import warnings

import numpy as np
import pandas as pd
import numba
import torch
import torchtuples as tt


def sample_alive_from_dates(dates, at_risk_dict, n_control=1):
    '''Sample index from living at time given in dates.
    dates: np.array of times (or pd.Series).
    at_risk_dict: dict with at_risk_dict[time] = <array with index of alive in X matrix>.
    n_control: number of samples.
    '''
    lengths = np.array([at_risk_dict[x].shape[0] for x in dates])  # Can be moved outside
    idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype('int')
    samp = np.empty((dates.size, n_control), dtype=int)
    samp.fill(np.nan)

    for it, time in enumerate(dates):
        samp[it, :] = at_risk_dict[time][idx[:, it]]
    return samp

def make_at_risk_dict(durations):
    """Create dict(duration: indices) from sorted df.
    A dict mapping durations to indices.
    For each time => index of all individual alive.
    
    Arguments:
        durations {np.arrary} -- durations.
    """
    assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep='first')
    at_risk_dict = dict()
    for ix, t in keys.iteritems():
        at_risk_dict[t] = allidx[ix:]
    return at_risk_dict


def make_at_risk_dict_with_starts(durations, starts):
    """Create dict(duration: indices) from sorted df.
    A dict mapping durations to indices.
    For each time => index of all individual alive.
    The vector `starts` indicates the "birthdate" of the subject (i.e. it is not at risk before this time).

    Arguments:
        durations {np.array} -- durations.
        starts {np.array} -- event starts.
    """
    assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep='first')
    at_risk_dict = dict()

    assert type(starts) is np.ndarray, 'Need starts to be a numpy array'
    start_ids = np.argsort(starts)
    starts = starts[start_ids]

    for i, (ix, t) in enumerate(keys.iteritems()):
        print(i, ix, t)
        at_risk_dict[t] = allidx[ix:]

        # remove individuals that were not born yet
        invalids = starts > t
        min_id = np.argmax(invalids)
        # no need to drop anything anymore
        if not invalids[min_id]:
            continue

        drop_ids = start_ids[min_id:]
        at_risk_dict[t] = np.setdiff1d(at_risk_dict[t], drop_ids)

    return at_risk_dict


class DurationSortedDataset(tt.data.DatasetTuple):
    """We assume the dataset contrain `(input, durations, events)`, and 
    sort the batch based on descending `durations`.

    See `torchtuples.data.DatasetTuple`.
    """
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tt.tuplefy(input, event).iloc[idx_sort]
        return batch


class CoxCCDataset(torch.utils.data.Dataset):
    def __init__(self, input, durations, events, n_control=1):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        self.at_risk_dict = make_at_risk_dict(durations)

        self.input = tt.tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if (not hasattr(index, '__iter__')) and (type(index) is not slice):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
        x_control = tt.TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
        return tt.tuplefy(x_case, x_control).to_tensor()

    def __len__(self):
        return len(self.durations)


class CoxTimeDataset(CoxCCDataset):
    def __init__(self, input, durations, events, n_control=1):
        super().__init__(input, durations, events, n_control)
        self.durations_tensor = tt.tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        durations = self.durations_tensor.iloc[index]
        case, control = super().__getitem__(index)
        case = case + durations
        control = control.apply_nrec(lambda x: x + durations)
        return tt.tuplefy(case, control)


class CoxVaccDataset(torch.utils.data.Dataset):
    def __init__(self, input, time_var_input, durations, events, starts, vaccmap, n_control=1, cached_dict=None,
                 min_dur=None, labtrans=None, return_weights=False):

        # events and durations
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        assert type(self.durations) is pd.Series

        self.durations_tensor = tt.tuplefy(self.durations.values.reshape(-1, 1))#.to_tensor()
        self.min_dur = min_dur

        # construct at risk dict
        if cached_dict is None:
            self.at_risk_dict = make_at_risk_dict_with_starts(durations, starts)
        else:
            self.at_risk_dict = cached_dict

        # input is separated to ordinary and time dependant variable
        self.input = tt.tuplefy(input)
        self.time_var_input = tt.tuplefy(time_var_input) if time_var_input is not None else None

        self.n_control = n_control
        self.labtrans=labtrans
        self.return_weights = return_weights

        # vacc related info
        self.starts = starts
        if vaccmap.dtype == np.bool:
            self.vaccmap = (~vaccmap).astype(int).to_numpy()
        else:
            warnings.warn("Passing vaccmap that does not have type bool.")
            self.vaccmap = vaccmap

    def __getitem__(self, index):
        if (not hasattr(index, '__iter__')) and (type(index) is not slice):
            index = [index]

        durations = self.durations_tensor.iloc[index]
        fails = self.durations.iloc[index]
        x_case, x_ctrl, case_starts, ctrl_starts, case_vacc, ctrl_vacc = self.get_case_control(fails, durations)

        # mask unvaccinated durations with a dummy value
        x_case = x_case + shift_duration(durations, case_starts[:, np.newaxis], case_vacc,
                                         min_dur=self.min_dur, labtrans=self.labtrans)
        x_ctrl = tt.TupleTree(x_ctrl[idx] + \
                              shift_duration(durations, ctrl_starts[idx][:, np.newaxis], ctrl_vacc[idx],
                                             min_dur=self.min_dur, labtrans=self.labtrans)
                              for idx in range(len(x_ctrl)))

        if not self.return_weights:
            return tt.tuplefy(x_case, x_ctrl).to_tensor()

        return tt.tuplefy(x_case, x_ctrl, durations[0][:, 0]).to_tensor()

    def get_case_control(self, fails, durs):
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)

        # get vacc info
        case_vacc, control_vacc = self.vaccmap[fails.index], self.vaccmap[control_idx.transpose()]
        case_starts, control_starts = self.starts[fails.index], self.starts[control_idx.transpose()]

        # sample x features of case/control, shift the time dependents
        x_case = self._integrate_time_vars(durs, fails.index, case_vacc)

        x_control = tt.TupleTree(
            self._integrate_time_vars(durs, idx, vacc)  #TODO je control vacc dobře
            for idx, vacc in zip(control_idx.transpose(), control_vacc)
        )

        return x_case, x_control, case_starts, control_starts, case_vacc, control_vacc

    def _integrate_time_vars(self, durs, idx, vaccmap):
        ordinary_vars = self.input.iloc[idx]
        if self.time_var_input is None:
            return ordinary_vars

        time_vars = self.time_var_input.iloc[idx]
        return combine_with_time_vars(ordinary_vars[0], time_vars[0], durs, vaccmap,
                                      min_dur=self.min_dur, labtrans=self.labtrans)

    def __len__(self):
        return len(self.durations)


def combine_with_time_vars(vars, time_vars, durs, vacc, min_dur=0.0, labtrans=None):
    time_vars = shift_duration(durs, time_vars, vacc, min_dur=min_dur, labtrans=labtrans)[0]
    return tt.tuplefy(np.hstack([vars, time_vars]))


def shift_duration(durs, starts, vacc, min_dur=0.0, labtrans=None):
    res = durs[0] - starts + min_dur

    vacc = np.nonzero(vacc)
    res[vacc] = min_dur
    res[res < 0] = min_dur  # mask durations in the future

    if labtrans is not None:
        res, _ = labtrans.transform(res, np.zeros((0,)))

    return tt.tuplefy(res.reshape(starts.shape))


@numba.njit
def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat


class DeepHitDataset(tt.data.DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)
