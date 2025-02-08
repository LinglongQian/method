import abc
from bisect import bisect
from typing import Sequence, Callable

import numpy as np
import torch as th

from ..special_tokens import SpecialToken


class TimeAwareDataset(th.utils.data.Dataset):
    """
    Enhanced Timeline Dataset with proper patient boundary handling and sequence generation.
    Supports:
    1. Proper patient context for each segment
    2. Cross-patient sequence construction
    3. Circular patient data access
    4. Valid sequence verification
    """

    BASE_YEAR = 1970

    def __init__(self, data: dict, encode: Callable, sequence_length: int = 2048):
        self.times: th.Tensor = data["times"]
        self.tokens: th.Tensor = data["tokens"]
        self.patient_context: th.Tensor = data["patient_context"]
        self.age_reference: Sequence[np.ndarray[np.float64]] = data["age_reference"]
        self.patient_offsets: np.ndarray[np.int64] = data["patient_data_offsets"]
        self.patient_ids: np.ndarray = data["patient_ids"]
        # if ids are bytes, convert them to string to be able to use .item() later
        if isinstance(self.patient_ids[0], bytes):
            self.patient_ids = self.patient_ids.astype("U")
        # +1 - token of age, +1 - token of datetime, both computed in runtime
        self.context_len: int = self.patient_context.shape[1] + 1 + 1
        self.timeline_len: int = sequence_length
        # vocab encode function that translates strings to integers
        self.encode: Callable = encode
        self.sep_token = self.encode(SpecialToken.PATIENT_SEP_TOKEN)
        self.context_times = th.zeros(self.context_len)

    def __len__(self) -> int:
        return len(self.times)
    
    def __getitem__(self, idx: int):
        """
        Get a training sample with proper patient context handling.
        
        Args:
            idx: Starting position in timeline
            
        Returns:
            - x: Input sequence with patient contexts
            - time_indices_x: Relative time indices for each patient in the input sequence
            - patient_indices: Patient indices for each part of the input sequence
            - y: Target sequence (shifted one position right or with patient separator)
            - time_indices_y: Relative time indices for each patient in the target sequence
        """
        return self._get_sequence_with_contexts(idx)

    def _get_patient_context(self, idx: int) -> th.Tensor:
        """Returns the patient context at the time given by the index."""
        patient_idx = self._get_patient_idx(idx)
        patient_context = self.patient_context[patient_idx]

        patient_age_at_timeline_start = self.times[idx]
        patient_age_token = self._years_to_token(patient_age_at_timeline_start)
        year = self._get_year_at_timeline_start(patient_idx, patient_age_at_timeline_start)
        anchor_year_token = self._years_to_token(year - self.BASE_YEAR)
        return th.cat((patient_context, th.tensor([patient_age_token, anchor_year_token])))

    def _get_patient_idx(self, idx: int) -> int:
        """Given the index in data, returns the patient's index (no.) in the patient data."""
        patient_idx = bisect(self.patient_offsets, idx)
        return patient_idx - 1

    def _years_to_token(self, years: float) -> int:
        return self.encode(SpecialToken.age_to_year_token(years))

    def _get_year_at_timeline_start(self, patient_idx: int, patient_age: float) -> float:
        year_of_birth = self.age_reference[patient_idx]
        year = year_of_birth + patient_age
        return year.item()

    def _get_relative_time_indices(self, timeline_times_x, timeline_times_y):
        # sort and get unique timestamps with relative indices
        combined_times = th.cat([timeline_times_x, timeline_times_y])
        _, combined_indices = th.unique(combined_times, sorted=True, return_inverse=True)
        relative_indices_x = combined_indices[:len(timeline_times_x)]
        relative_indices_y = combined_indices[len(timeline_times_y):]
        return relative_indices_x, relative_indices_y

    def _process_segment(self, start_pos, end_pos, patient_end: bool = False):
        """Process a segment of the timeline.
        
        Args:
            start_pos: Start position in the timeline
            end_pos: End position in the timeline
            patient_end: Whether this is the last segment for current patient
            
        Returns:
            Tuple of (x_sequence, y_sequence, time_indices_x, time_indices_y, patient_indices)
        """
        # Get main sequence
        # add patient context to the sequence
        patient_context = self._get_patient_context(start_pos)
        x_sequence = th.cat([patient_context, self.tokens[start_pos : end_pos]])
        if patient_end:
            # for the last segment, add the patient separator token
            y_sequence = th.cat([
                patient_context, 
                self.tokens[start_pos + 1 : end_pos], 
                th.tensor([self.sep_token]),
                ])
        else:
            # for other segments, shift the sequence by one
            y_sequence = th.cat([
                patient_context, 
                self.tokens[start_pos + 1 : end_pos + 1],
                ])

        # generate sequence time
        timeline_times_x = th.cat([
                self.context_times, 
                self.times[start_pos : end_pos],
                ])
        if patient_end:
            # for the last segment, duplicate the last timestamp as patient separator time
            timeline_times_y = th.cat([
                self.context_times,
                self.times[start_pos + 1 : end_pos], 
                self.times[end_pos-1].unsqueeze(0),
                ])
        else:
            # for other segments, shift the timeline by one
            timeline_times_y = th.cat([
                self.context_times,
                self.times[start_pos + 1 : end_pos + 1],
                ])

        # generate sequence relative time indices
        x_time_indices, y_time_indices = self._get_relative_time_indices(timeline_times_x, timeline_times_y)
        # assert ((y_time_indices - x_time_indices).max() <= 1) & ((y_time_indices - x_time_indices).min() >= 0), f"Invalid time indices, {start_pos}, {end_pos}"
        return x_sequence, y_sequence, x_time_indices, y_time_indices

    def _get_sequence_with_contexts(self, idx: int):
        sequences_x = []
        sequences_y = []
        time_indices_x = []
        time_indices_y = []
        patient_indices = []
        
        total_length = 0
        current_pos = idx
        invalid_indices = []
        # get the first patient index
        patient_idx = self._get_patient_idx(current_pos)

        while total_length < self.timeline_len:
            # get current patient end position
            patient_end_pos = (self.patient_offsets[patient_idx + 1] 
                            if patient_idx + 1 < len(self.patient_offsets)
                            else len(self.tokens))
            if patient_end_pos == current_pos:
                # Skip patients with no tokens
                patient_idx = (patient_idx + 1) % len(self.patient_offsets)
                current_pos = self.patient_offsets[patient_idx]
                continue

            # calculate available timeline length to ensure we don't exceed total length
            remaining_length = self.timeline_len - total_length
            available_in_current = patient_end_pos - current_pos + self.context_len
            try:
                if available_in_current > remaining_length:
                    # If the current patient has sufficient tokens, process the segment
                    x_sequence, y_sequence, x_time_indices, y_time_indices = self._process_segment(
                                                                                        current_pos, 
                                                                                        current_pos + remaining_length - self.context_len,
                                                                                        patient_end=False
                                                                                        )
                    time_diff_max = (y_time_indices - x_time_indices).max()
                    time_diff_min = (y_time_indices - x_time_indices).min()
                    if time_diff_max > 1 or time_diff_min < 0:
                        invalid_indices.append(('Not patient end',
                                                total_length,
                                                current_pos, current_pos + remaining_length - self.context_len, 
                                                time_diff_max.item(), time_diff_min.item()))
                        current_pos += 1
                        continue

                    sequences_x.append(x_sequence)
                    sequences_y.append(y_sequence)
                    time_indices_x.append(x_time_indices)
                    time_indices_y.append(y_time_indices)
                    patient_indices.append(th.full(x_sequence.size(), total_length))
                    total_length += len(x_sequence)
                    break

                # Otherwise, process the available tokens of the current patient and move to the next patient tokens
                x_sequence, y_sequence, x_time_indices, y_time_indices = self._process_segment(
                                                                                        current_pos, 
                                                                                        patient_end_pos, 
                                                                                        patient_end=True
                                                                                        )
                time_diff_max = (y_time_indices - x_time_indices).max()
                time_diff_min = (y_time_indices - x_time_indices).min()
                if time_diff_max > 1 or time_diff_min < 0:
                    invalid_indices.append(('Patient end',
                                            total_length,
                                            current_pos, patient_end_pos, 
                                            time_diff_max.item(), time_diff_min.item()))
                    current_pos += 1
                    continue
                                                                                    
                sequences_x.append(x_sequence)
                sequences_y.append(y_sequence)
                time_indices_x.append(x_time_indices)
                time_indices_y.append(y_time_indices)
                patient_indices.append(th.full(x_sequence.size(), total_length))
                total_length += len(x_sequence)
                
                # update patient index and current position
                patient_idx = (patient_idx + 1) % len(self.patient_offsets)
                current_pos = self.patient_offsets[patient_idx]

            except Exception as e:
                # print(f"Error at {current_pos}, {patient_end_pos}, {remaining_length}, {total_length}: {str(e)}")
                current_pos += 1
                continue
        
        # if invalid_indices:
        #     print(f"Foud {len(invalid_indices)} invalid indices:")
        #     for invalid_status, length, start, end, max_diff, min_diff in invalid_indices:
        #         print(f"{invalid_status}, current length: {length}, position: {start} - {end}: max_diff={max_diff}, min_diff={min_diff}")

        # Ensure the sequence ends with a valid token
        if total_length > self.timeline_len:
            return self._get_sequence_with_contexts(idx+1)
    
        x = th.cat(sequences_x, dim=0)
        y = th.cat(sequences_y, dim=0)
        time_x = th.cat(time_indices_x)
        time_y = th.cat(time_indices_y)
        patient_indices = th.cat(patient_indices)

        assert len(x) == len(y) == len(time_x) == len(time_y) == len(patient_indices) == self.timeline_len, \
            f"Invalid sequence length: {len(x)}"

        return x, time_x, patient_indices, y, time_y
        

class InferenceDataset(TimeAwareDataset, abc.ABC):
    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx) -> tuple[th.Tensor, dict]:
        pass

    def _get_indices_of_stokens(self, stokens: str | Sequence[str]) -> np.ndarray[np.int64]:
        tokens = self.encode(stokens)
        if np.isnan(tokens).any():
            raise ValueError(f"Tokens for {stokens} could not be found in the vocabulary.")
        tokens = th.tensor(tokens)
        return th.nonzero(th.isin(self.tokens, tokens)).view(-1).numpy()

    @staticmethod
    def _match_next_value(
        to_match: Sequence, match_with: Sequence, always_match: bool = True
    ) -> np.ndarray[int | float]:
        """
        Return the next closest values in `match_with` for every corresponding value in `to_match`.
        Both sequences must be sorted in the ascending order.

        If `always_match` is True, the function will always try to assign a value in `match_with`,
        if it does not find it, it will raise out-of-bounds error.
        If `always_match` is False, the function will return `np.nan` for every value without the
        match.
        """
        match_with_indices = np.searchsorted(match_with, to_match)
        if always_match:
            return match_with[match_with_indices]
        else:
            matched_values = np.fromiter(
                (
                    match_with[match_with_idx] if match_with_idx < len(match_with) else np.nan
                    for match_with_idx in match_with_indices
                ),
                dtype=float,
                count=len(to_match),
            )
            if not np.isnan(matched_values[-1]):
                return matched_values.astype(int)
            return matched_values
