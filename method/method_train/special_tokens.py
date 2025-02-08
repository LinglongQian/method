from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .constants import SEPARATOR_DTYPE


class SpecialToken:
    PATIENT_SEP_TOKEN = "_PATIENT_SEP"

    SEPARATORS = {
        # k: v / (60 * 24 * 365.25)
        k: np.timedelta64(v, 'm')  # convert to minutes
        for k, v in {
            "_5m-15m": 5,
            "_15m-1h": 15,
            "_1h-2h": 60,
            "_2h-6h": 2 * 60,
            "_6h-12h": 6 * 60,
            "_12h-1d": 12 * 60,
            "_1d-3d": 24 * 60,
            "_3d-1w": 3 * 24 * 60,
            "_1w-2w": 7 * 24 * 60,
            "_2w-1mt": 2 * 7 * 24 * 60,
            "_1mt-3mt": 30 * 24 * 60,
            "_3mt-6mt": 3 * 30 * 24 * 60,
            "_=6mt": 6 * 30 * 24 * 60,
        }.items()
    }
    SEPARATOR_NAMES = list(SEPARATORS.keys())
    SEPARATOR_SIZES = np.fromiter(SEPARATORS.values(), dtype=SEPARATOR_DTYPE)
    TIMELINE_END = "_TIMELINE_END"
    DEATH = "_DEATH"
    STATES_UNKNOWN = "_STATES_UNKNOWN"
    GRADIENTS_UNKNOWN = "_GRADIENTS_UNKNOWN"
    DURATION_UNKNOWN = "_DURATION_UNKNOWN"

    # e.g., used for defining age of a person
    YEARS = {"_<5": 5, **{f"_{i - 5}-{i}y": i for i in range(10, 101, 5)}}
    YEAR_NAMES = list(YEARS.keys()) + ["_>100"]
    YEAR_BINS = list(YEARS.values())

    DECILES = [f"_Q{i}" for i in range(1, 11)]

    # State tokens
    STATE_HIGH = "_HIGH"
    STATE_LOW = "_LOW"
    STATE_NORMAL = "_NORMAL"
    STATES = [STATE_LOW, STATE_HIGH, STATE_NORMAL]

    # Gradient tokens
    GRADIENT_INCREASING = "_INCREASING"
    GRADIENT_DECREASING = "_DECREASING"
    GRADIENT_CONSTANT = "_CONSTANT"
    GRADIENTS = [GRADIENT_CONSTANT, GRADIENT_INCREASING, GRADIENT_DECREASING]

    # Duration tokens
    DURATION_LONG = "_LONG"
    DURATION_SHORT = "_SHORT"
    DURATION_MEDIUM = "_MEDIUM"
    DURATIONS = [DURATION_LONG, DURATION_SHORT, DURATION_MEDIUM]

    # Combined list of all special tokens
    ALL = [
        PATIENT_SEP_TOKEN,  
        *SEPARATOR_NAMES,
        TIMELINE_END,
        DEATH,
        STATES_UNKNOWN,
        GRADIENTS_UNKNOWN,
        DURATION_UNKNOWN,
        *YEAR_NAMES,
        *DECILES,
        *STATES,
        *GRADIENTS,
        *DURATIONS,
    ]

    @staticmethod
    def get_longest_separator() -> (str, float):
        return SpecialToken.SEPARATOR_NAMES[-1], SpecialToken.SEPARATOR_SIZES[-1]

    @staticmethod
    def age_to_year_token(age: float) -> str:
        i = np.digitize(age, SpecialToken.YEAR_BINS)
        return SpecialToken.YEAR_NAMES[i]
        