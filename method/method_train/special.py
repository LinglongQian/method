from typing import Optional

import numpy as np
import pandas as pd

from ..base import ContextData, SimpleData
from ..constants import DataProp, SECONDS_IN_YEAR


class MimicPatientBirthDateData(ContextData):
    COLUMNS_WE_USE = ["anchor_age", "anchor_year"]

    def __init__(self, data_prop, **kwargs):
        super().__init__("hosp/patients", data_prop, use_cols=self.COLUMNS_WE_USE, **kwargs)

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df["anchor_year"] -= df["anchor_age"]
        df.drop(columns="anchor_age", inplace=True)
        df.anchor_year = pd.to_datetime(df.anchor_year, format="%Y")
        df.anchor_year += pd.DateOffset(months=6)
        return df


class ICUStayIdMixin:
    def __init__(self, patient_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_prop = DataProp.create("mimiciv", "test")
        self.icu_stay_df = SimpleData(
            "icu/icustays",
            data_prop,
            use_cols=["stay_id", "intime"],
            parse_dates=["intime"],
        ).df
        patient_order = {pid: i for i, pid in enumerate(patient_ids)}
        self.icu_stay_df["patient_order"] = self.icu_stay_df.subject_id.map(patient_order)
        self.icu_stay_df.sort_values(["patient_order", "intime"], inplace=True)

    def _get_stay_id(self, patient_id: int):
        if patient_id not in self.icu_stay_df.subject_id.values:
            return None
        return self.icu_stay_df.loc[self.icu_stay_df.subject_id == patient_id].stay_id.values


class HadmIdMixin:
    _inpatient_stay: Optional[np.ndarray]

    def __init__(self, patient_ids=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if patient_ids is None:
            self._inpatient_stay = None
        else:
            data_prop = DataProp.create("mimiciv", "test")
            inpatient_stay_df = SimpleData(
                "hosp/admissions",
                data_prop,
                use_cols=["hadm_id", "admittime"],
                parse_dates=["admittime"],
            ).df
            patient_order = {pid: i for i, pid in enumerate(patient_ids)}
            inpatient_stay_df["patient_order"] = inpatient_stay_df.subject_id.map(patient_order)
            inpatient_stay_df.sort_values(["patient_order", "admittime"], inplace=True)
            self._inpatient_stay = inpatient_stay_df.hadm_id.values

    def _get_hadm_id(self, admission_no: int):
        return self._inpatient_stay[admission_no]
