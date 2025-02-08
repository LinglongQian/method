from collections import defaultdict
from typing import Callable
from torch.nn import functional as F
import numpy as np
import pandas as pd
import torch as th
from ..utils import log_memory
from ..method_tokenize import SpecialToken

@th.no_grad()
def Timeaware_estimate_loss(model, ctx, get_batch: Callable, eval_iters: int, tokens_of_interest: dict, sliding_window_size: int):
    if hasattr(model, "config"):
        sequence_length = model.config.sequence_length
        timeaware = model.config.timeaware
    else:
        sequence_length = model.module.config.sequence_length
        timeaware = model.module.config.timeaware

    block_thresh = sequence_length // 2
    out = {}
    model.eval()
    for split in ["train", "val"]:
        # log_memory(f'Before evaluation {split}')
        losses = th.empty(eval_iters)
        all_tokens_res = defaultdict(list)
        toi_res = defaultdict(list)
        time_correct = 0
        time_total = 1e-10
        all_uncertain = th.empty(eval_iters)
        for i in range(eval_iters):
            X, time_indices_x, patient_indices, Y, time_indices_y = get_batch(split)
            with ctx:
                outputs = model(
                                inputs=X,
                                time_indices=time_indices_x,
                                patient_indices=patient_indices, 
                                target=Y,
                                target_time_indices=time_indices_y,
                                sliding_window_size=sliding_window_size
                                )

            loss = outputs['loss']
            logits = outputs['logits']
            if timeaware:
                time_logits = outputs['time_logits']

            losses[i] = loss.item()
            all_uncertain[i] = get_prediction_uncertainty(logits).mean()  # [B, T]
            if timeaware:
                time_labels = time_indices_y - time_indices_x
                time_probs = (th.sigmoid(time_logits.squeeze().squeeze()) > 0.5).long()
                valid_mask = (time_indices_y != 0)
                time_correct += (time_probs[valid_mask] == time_labels[valid_mask]).sum().item()
                time_total += valid_mask.sum()
                all_uncertain[i] = get_prediction_uncertainty(logits, time_logits).mean()

            if split == "val":
                for k in [1, 3, 5]:
                    all_tokens_res[f"acc_top/all/{split}/{block_thresh}/k={k}"].append(
                        top_k_accuracy(logits, Y, k=k, threshold=block_thresh)
                    )
                for stoken, token in tokens_of_interest.items():
                    for k in [1, 3, 10]:
                        toi_res[f"acc_top/{stoken}/{split}/{block_thresh}/k={k}"].append(
                            top_k_acc_special(logits, Y, token, k=k, threshold=block_thresh)
                        )

        if timeaware:
            out[f"time_acc/{split}"] = time_correct / time_total
        out[f"uncertain/{split}"] = all_uncertain.mean()
        out[f"loss/{split}"] = losses.mean()
        out.update({test_name: np.mean(v) for test_name, v in all_tokens_res.items()})
        out.update({test_name: compute_weighted_mean(v) for test_name, v in toi_res.items()})
        # log_memory(f'After evaluation {split}')
    model.train()
    return out


def get_prediction_uncertainty(token_logits, time_logits=None):
    """quantify the uncertainty of model predictions
    
    Args:
        time_logits: [B, T, 1]
        token_logits: [B, T, vocab_size]
        
    Returns:
        uncertainty: [B, T] uncertainty score for each token prediction
    """
    # 1. uncertainty of token prediction
    token_probs = F.softmax(token_logits, dim=-1)  # [B, T, vocab_size]
    token_entropy = -(token_probs * th.log(token_probs + 1e-10)).sum(dim=-1)  # [B, T]
    # normalize entropy to [0, 1]
    uncertainty = token_entropy / np.log(token_logits.size(-1))

    if time_logits is not None:
        # 2. uncertainty of time prediction
        time_probs = th.sigmoid(time_logits.squeeze(-1))  # [B, T]
        time_uncertainty = 1 - 2 * th.abs(time_probs - 0.5)  # more uncertain when closer to 0.5

        # 3. combine time and token uncertainty
        # when time and token predictions are both uncertain, the overall uncertainty is higher
        uncertainty = (time_uncertainty + uncertainty) / 2
    
    return uncertainty


def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return th.nan
    return sum(x[0] * x[1] for x in v) / weights


def top_k_accuracy(logits, y, k, threshold):
    # logits: (batch_size, block_size, vocab_size)
    logits = logits[:, threshold - 1 :, :]
    if y.dim() == 1:
        y_true = y[threshold - 1 :].unsqueeze(0).unsqueeze(-1)
    else:
        y_true = y[:, threshold - 1 :, None]
    k = min(k, logits.size(-1))
    _, indices = th.topk(logits, k)
    correct = (indices == y_true).any(dim=-1)
    return correct.sum().item() / correct.numel()


def top_k_acc_special(logits, y, token_of_interest, k: int, threshold: int):
    logits = logits[:, threshold - 1 :, :]
    if y.dim() == 1:
        y_true = y[threshold - 1 :].unsqueeze(0).unsqueeze(-1)
    else:
        y_true = y[:, threshold - 1 :, None]
    interested = y_true == token_of_interest
    if interested.sum() == 0:
        return 0, 0
    _, indices = th.topk(logits, min(k, logits.size(-1)))
    correct = (indices == y_true).any(dim=-1, keepdim=True)
    weight = interested.sum()
    score = (correct & interested).sum() / weight
    return score.item(), weight.item()


def process_admission_results(filename: str, discharge_stoken: str) -> pd.DataFrame:
    res_dir = PROJECT_ROOT / "results" / filename
    df = pd.concat(pd.read_json(res_path) for res_path in res_dir.iterdir())
    df.rename(
        columns={
            "actual": "actual_token",
            "expected": "expected_token",
            "patient_id": "subject_id",
        },
        inplace=True,
    )
    prev_len = len(df)
    df = df.loc[df.actual_token.isin([discharge_stoken, SpecialToken.DEATH])]
    print(
        "Dropped rows due to an ambiguous result: {:,}/{:,} ({:.3%})".format(
            prev_len - len(df), prev_len, (prev_len - len(df)) / prev_len
        )
    )
    df["actual"] = (df.actual_token == SpecialToken.DEATH).astype(int)
    df["expected"] = (df.expected_token == SpecialToken.DEATH).astype(int)
    df_gb = df.groupby("admission_token_idx", dropna=False)
    agg_scheme = {
        "subject_id": "first",
        "expected": "first",
        "actual": "mean",
        "true_token_time": "first",
        "true_token_dist": "first",
        "token_time": "mean",
        "token_dist": "mean",
        "patient_age": "first",
    }
    if "stay_id" in df.columns:
        agg_scheme["stay_id"] = "first"

    def ci(df_: pd.DataFrame, sigmas: int = 2) -> bool:
        mean, std = df_.token_time.aggregate(["mean", "std"])
        true_time = df_.true_token_time.iloc[0]
        return abs(mean - true_time) < std * sigmas

    return (
        df_gb.agg(agg_scheme)
        .join(df_gb.agg(count=("actual", "count"), token_time_std=("token_time", "std")))
        .join(df_gb.apply(ci).rename("ci_2sig"))
        .join(df_gb.apply(partial(ci, sigmas=1)).rename("ci_1sig"))
        .reset_index(drop=True)
        .set_index("subject_id")
    )