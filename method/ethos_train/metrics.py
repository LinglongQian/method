from collections import defaultdict
from typing import Callable
from torch.nn import functional as F
import pandas as pd
import numpy as np
import torch as th
from ..ethos_tokenize import SpecialToken

@th.no_grad()
def estimate_loss(model, ctx, get_batch: Callable, eval_iters: int, tokens_of_interest: dict):
    if hasattr(model, "config"):
        block_size = model.config.block_size
    else:
        block_size = model.module.config.block_size
    block_thresh = block_size // 2
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = th.empty(eval_iters)
        all_uncertain = th.empty(eval_iters)
        all_tokens_res = defaultdict(list)
        toi_res = defaultdict(list)

        for i in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[i] = loss.item()
            all_uncertain[i] = get_prediction_uncertainty(logits).mean()
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
        out[f"loss/{split}"] = losses.mean()
        out[f"uncertainty/{split}"] = all_uncertain.mean()
        out.update({test_name: np.mean(v) for test_name, v in all_tokens_res.items()})
        out.update({test_name: compute_weighted_mean(v) for test_name, v in toi_res.items()})
    model.train()
    return out


def get_prediction_uncertainty(token_logits):
    token_probs = F.softmax(token_logits, dim=-1)  # [B, T, vocab_size]
    token_entropy = -(token_probs * th.log(token_probs + 1e-10)).sum(dim=-1)  # [B, T]
    uncertainty = token_entropy / np.log(token_logits.size(-1))

    return uncertainty

    
def compute_weighted_mean(v):
    weights = sum(x[1] for x in v)
    if weights == 0:
        return th.nan
    return sum(x[0] * x[1] for x in v) / weights


def top_k_accuracy(logits, y, k, threshold):
    # logits: (batch_size, block_size, vocab_size)
    logits = logits[:, threshold - 1 :, :]
    y_true = y[:, threshold - 1 :, None]
    k = min(k, logits.size(-1))
    _, indices = th.topk(logits, k)
    correct = (indices == y_true).any(dim=-1)
    return correct.sum().item() / correct.numel()


def top_k_acc_special(logits, y, token_of_interest, k: int, threshold: int):
    logits = logits[:, threshold - 1 :, :]
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