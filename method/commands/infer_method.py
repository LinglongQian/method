from functools import partial
from multiprocessing import set_start_method
from pathlib import Path
from typing import Optional

import numpy as np
from click import command, option, Choice
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Subset

from method.constants import (
    ADMISSION_STOKEN,
    DISCHARGE_STOKEN,
    PROJECT_DATA,
    ICU_DISCHARGE_STOKEN,
    ICU_ADMISSION_STOKEN,
)
from method.method_train.datasets import (
    ReadmissionDataset,
    AdmissionMortalityDataset,
    MortalityDataset,
    SingleAdmissionMortalityDataset,
    SofaPredictionDataset,
    ICUMortalityDataset,
)
from method.method_train.datasets.mimiciv import DrgPredictionDataset, ICUReadmissionDataset
from method.method_train.utils import load_model_from_checkpoint
from method.method_inference import Test, run_inference
from method.utils import load_data, get_logger

logger = get_logger()


@command
@option("--test", required=True, type=Choice([t.value for t in Test]))
@option("--model", required=True, help="Path to the model checkpoint.")
@option("--data", required=True, help="Path of the tokenized dataset in PROJECT_DATA.")
@option("--vocab", required=True, help="Path of the vocabulary in PROJECT_DATA.")
@option(
    "-n",
    "--n_tokens",
    type=int,
    default=None,
    help="Number of tokens in millions to load from the test data, `None` for all.",
)
@option("-j", "--n_jobs", type=int, default=1)
@option("-g", "--n_gpus", type=int, default=1)
@option("--suffix", type=str, default=None)
@option("--device", default="cuda", type=Choice(["cuda", "cpu"]))
@option("--no_compile", is_flag=True, help="Don't compile the model using Triton.")
@option("--output", default="results", help="Path where to save results.")
@option("--model_name", default=None, help="Name of the model, used for the output directory.")
@option("--no_time_offset", is_flag=True, help="Don't do 24h-time-offset for ICU mortality.")
@option("--data_type", type=Choice(["ethos", "method"]), default="ethos")
@option("--wta", is_flag=True, help="Load time-agnostic model.")
@option("--infer_sequence_length", type=int, default=-1)

def infer_method(
    test: str,
    model: str,
    data: str,
    vocab: str,
    n_tokens: Optional[int],
    n_jobs: int,
    n_gpus: int,
    suffix: str,
    device: str,
    no_compile: bool,
    output: str,
    model_name: Optional[str],
    no_time_offset: bool,
    data_type: str,
    wta: bool,
    infer_sequence_length: int,
):
    if data_type == "method":
        from method.method_tokenize import SpecialToken, QITVocabulary
        vocab = QITVocabulary(PROJECT_DATA / vocab)
    else:
        from method.ethos_tokenize import SpecialToken, Vocabulary
        vocab = Vocabulary(PROJECT_DATA / vocab)
        PATIENT_SEP_TOKEN = "_PATIENT_SEP"
        vocab.tokenize(PATIENT_SEP_TOKEN)
        
    test = Test(test)
    stoi = [SpecialToken.DEATH, SpecialToken.TIMELINE_END]
    if test == Test.READMISSION:
        dataset_cls = ReadmissionDataset
        stoi = [ADMISSION_STOKEN] + stoi
    elif test == Test.ADMISSION_MORTALITY:
        dataset_cls = AdmissionMortalityDataset
        stoi = [DISCHARGE_STOKEN] + stoi
    elif test == Test.MORTALITY:
        dataset_cls = MortalityDataset
    elif test == Test.SINGLE_ADMISSION:
        # todo: move the hardcoded values to options of the script
        dataset_cls = partial(SingleAdmissionMortalityDataset, admission_idx=70885993, num_reps=35)
        stoi = [DISCHARGE_STOKEN] + stoi
    elif test == Test.SOFA_PREDICTION:
        dataset_cls = SofaPredictionDataset
        stoi += SpecialToken.DECILES
    elif test == Test.ICU_MORTALITY:
        dataset_cls = partial(ICUMortalityDataset, use_time_offset=not no_time_offset)
        stoi = [ICU_DISCHARGE_STOKEN] + stoi
    elif test == Test.DRG_PREDICTION:
        dataset_cls = DrgPredictionDataset
        drg_stokens = list(vocab.get_q_storage("DRG_CODE").values())
        assert drg_stokens, "No DRG stokens found in the vocabulary"
        stoi += drg_stokens
    elif test == Test.ICU_READMISSION:
        dataset_cls = ICUReadmissionDataset
        stoi = [ICU_ADMISSION_STOKEN, DISCHARGE_STOKEN] + stoi
    else:
        raise ValueError(f"Unknown test: {test}, available")

    model_path = Path(model)
    model, sequence_length = load_model_from_checkpoint(model_path, device, wta, for_training=False)
    logger.info(f"Model loaded (sequence_length={sequence_length})")

    if infer_sequence_length == -1:
        infer_sequence_length = sequence_length

    data_path = PROJECT_DATA / data
    # fold = data_path.stem.split("_")[1]
    model_name = model_path.stem if model_name is None else model_name
    token_suffix = f"_{n_tokens}M_tokens" if n_tokens is not None else ""
    results_dir = Path(output) / f"{test.value}_{model_name}{token_suffix}_infer_length_{infer_sequence_length}"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path, n_tokens=n_tokens * 1_000_000 if n_tokens is not None else None)
    data["times"].share_memory_()
    data["tokens"].share_memory_()
    data["patient_context"].share_memory_()

    logger.info(f"Inference sequence length: {infer_sequence_length}")
    dataset = dataset_cls(data=data, encode=vocab.encode, sequence_length=infer_sequence_length)
    logger.info(f"Dataset size: {len(dataset):,}")

    data = model, device, vocab, stoi, results_dir, test, suffix, no_compile
    indices = np.arange(len(dataset))
    subsets = (
        Subset(dataset, subset_indices) for subset_indices in np.array_split(indices, n_jobs)
    )
    loaders = [
        DataLoader(
            subset,
            batch_size=None,
            pin_memory=device == "cuda",
            batch_sampler=None,
            pin_memory_device=f"{device}:{i % n_gpus}" if device == "cuda" else "",
        )
        for i, subset in enumerate(subsets)
    ]
    set_start_method("spawn")
    Parallel(n_jobs=n_jobs)(delayed(run_inference)(loader, data, n_gpus) for loader in loaders)
    logger.info(f"Done, results saved to '{results_dir}'")