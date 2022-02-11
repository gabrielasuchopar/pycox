import numpy as np
import torch
import torchtuples as tt
from pycox import models
from pycox.models.data import combine_with_time_vars, shift_duration, de_tupletree, add_case_counts
from pycox.models.loss import TimeWeightedCoxCCLoss


class CoxVacc(models.cox_time.CoxTime):
    make_dataset = models.data.CoxVaccDataset

    def __init__(self, net, optimizer=None, device=None, shrink=0., labtrans=None, loss=None,
                 train_dict=None, val_dict=None, min_duration=None, weights=None, case_count_dict=None):
        if loss is None and weights is not None:
            loss = TimeWeightedCoxCCLoss(weights, shrink=shrink, device=device)

        super().__init__(net, optimizer=optimizer, device=device, shrink=shrink, labtrans=labtrans, loss=loss)
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.training_data = None
        self.min_duration = min_duration
        self.weighted = weights is not None

        self.case_count_dict = case_count_dict

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            n_control=1, shrink=None, time_var_input=None, val_time_var_input=None, **kwargs):

        input = input, time_var_input
        if val_data is not None:
            val_i, val_t = val_data
            val_i = val_i, val_time_var_input
            val_data = val_i, val_t

        # get sorted input data
        idx_sort = self._get_sort_idx(self.split_target_starts(target)[0])
        input, target = self._sorted_input_target(input, target, idx_sort=idx_sort)
        self.training_data = tt.tuplefy(input, target)

        durations, _ = self.split_target_starts(target)[0]
        self.min_duration = durations.min().astype(np.float32)

        return super().fit(input, target, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
            num_workers=num_workers, shuffle=shuffle, metrics=metrics, val_data=val_data, val_batch_size=val_batch_size,
            n_control=n_control, shrink=shrink, **kwargs)

    @staticmethod
    def split_target_starts(target):
        # get starts from target if applicable
        durations, events, starts, vaccmap, is_val = target
        target = durations, events
        return target, starts, vaccmap, is_val

    def target_to_df(self, target):
        target, _, _, _ = self.split_target_starts(target)
        return super().target_to_df(target)

    @staticmethod
    def _sorted_input_target(input, target, idx_sort=None):
        if idx_sort is None:
            return input, target

        sort_a_tupletree = lambda t: tt.tuplefy(t).iloc[idx_sort][0]

        target = [sort_a_tupletree(t) for t in target[:-1]] + [target[-1]]
        input, time_var_input = input
        input = sort_a_tupletree(input)
        time_var_input = sort_a_tupletree(time_var_input) if time_var_input is not None else None
        return (input, time_var_input), target

    def make_dataloader(self, data, batch_size, shuffle=True, num_workers=0, n_control=1):
        """Dataloader for training. Data is on the form (input, target), where
        target is (durations, events, start times, vaccmap, is_validation). Vaccmap is True if the subject has any
        immunity (vaccination or postinfection). Input has the form (variables, time dependent variables).

        Arguments:
            data {tuple} -- Tuple containing (input, target).
            batch_size {int} -- Batch size.

        Keyword Arguments:
            shuffle {bool} -- If shuffle in dataloader (default: {True})
            num_workers {int} -- Number of workers in dataloader. (default: {0})
            n_control {int} -- Number of control samples in dataloader (default: {1})

        Returns:
            dataloader -- Dataloader for training.
        """
        input, target = data
        idx_sort = self._get_sort_idx(self.split_target_starts(target)[0])
        input, target = self._sorted_input_target(input, target, idx_sort=idx_sort)

        # split tuples to components
        input, time_var_input = input
        target, starts, vaccmap, is_val = self.split_target_starts(target)
        durations, events = target

        if is_val:
            saved_dict = self.val_dict
        else:
            saved_dict = self.train_dict

        dataset = self.make_dataset(input, time_var_input, durations, events, starts, vaccmap, n_control=n_control,
                                    cached_dict=saved_dict, min_dur=self.min_duration, labtrans=self.labtrans,
                                    return_weights=self.weighted, case_count_dict=self.case_count_dict)

        dataloader = tt.data.DataLoaderBatch(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
        return dataloader


    @staticmethod
    def _get_sort_idx(target):
        target = tt.tuplefy(target).to_numpy()
        durations, _ = target

        idx_sort = np.argsort(durations)
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            return None

        return idx_sort

    def predict(self, input, starts, vaccmap, time_var_input=None, time_is_real_time=False, **kwargs):
        input, time = input
        starts = starts if len(starts.shape) > 1 else starts[:, np.newaxis]
        ref_duration = (starts + time) if not time_is_real_time else time

        if self.case_count_dict is not None:
            input = add_case_counts(ref_duration, de_tupletree(input), self.case_count_dict)

        if time_var_input is not None:
            vaccmap = (~vaccmap).astype(int).to_numpy()
            input = combine_with_time_vars(de_tupletree(input), time_var_input, ref_duration, vaccmap,
                                           min_dur=self.min_duration, labtrans=self.labtrans)

        if time_is_real_time:
            time = shift_duration(time, starts, vaccmap, min_dur=self.min_duration, labtrans=self.labtrans)
            time = de_tupletree(time)
        else:
            time += self.min_duration
            vacc = np.nonzero(vaccmap)
            time[vacc] = self.min_duration
            time, _ = self.labtrans.transform(time, np.zeros((0,)))
            time = time[:, np.newaxis]

        return super().predict((input, time), **kwargs)

    def compute_metrics(self, input, metrics):
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        input = self._to_device(input)
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same length.")
        if self.weighted:
            case, control, weights = input
        else:
            case, control = input  # both are TupleTree

        input_all = tt.TupleTree((case,) + control).cat()
        g_all = self.net(*input_all)
        g_all = tt.tuplefy(g_all).split(batch_size).flatten()
        g_case = g_all[0]
        g_control = g_all[1:]
        args = (g_case, g_control, weights) if self.weighted else (g_case, g_control)
        res = {name: metric(*args) for name, metric in metrics.items()}
        return res
