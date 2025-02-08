import bisect
import math
from .model import ModelConfig, Ethos
from ..utils import load_data
import torch as th

def make_infinite_loader(loader):
    while True:
        for batch in iter(loader):
            yield batch


def get_lr(it, args):
    """Learning rate decay scheduler (cosine with warmup)."""
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.lr * it / args.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.lr - args.min_lr)


def get_train_val_data(data_path, val_frac: float) -> tuple[dict, dict]:
    times, tokens, patient_context, age_reference, patient_data_offsets, patient_ids = load_data(
        data_path
    ).values()
    # get N last patients for validation dataset
    val_size = round(len(times) * val_frac)
    # we don't want to do it in the middle of a timeline, so we find the last timeline starting
    # point before the index len(times) - val_size
    idx = bisect.bisect(patient_data_offsets, len(times) - val_size) - 1
    split_data_idx = patient_data_offsets[idx]

    tokens_train = tokens[:split_data_idx].clone()
    tokens_val = tokens[split_data_idx:].clone()
    del tokens

    times_train = times[:split_data_idx].clone()
    times_val = times[split_data_idx:].clone()
    del times

    train_data = {
        "tokens": tokens_train,
        "times": times_train,
        "patient_context": patient_context,
        "age_reference": age_reference,
        "patient_data_offsets": patient_data_offsets[:idx],
        "patient_ids": patient_ids,
    }
    val_data = {
        "tokens": tokens_val,
        "times": times_val,
        "patient_context": patient_context,
        "age_reference": age_reference,
        "patient_data_offsets": [x - split_data_idx for x in patient_data_offsets[idx:]],
        "patient_ids": patient_ids,
    }
    return train_data, val_data


def load_model_from_checkpoint(path, device, for_training=True, **kwargs):
    checkpoint = th.load(path, map_location=device, weights_only=False)
    gptconf = ModelConfig(**checkpoint["model_args"])
    model = Ethos(gptconf, **kwargs)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if not for_training:
        return model, checkpoint["config"].block_size
    best_val_loss = checkpoint.get("best_val_loss", 1e9)
    return model, checkpoint["iter_num"], best_val_loss, checkpoint["optimizer"]