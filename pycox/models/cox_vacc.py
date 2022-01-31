import numpy as np
import torchtuples as tt
from pycox import models


class CoxVacc(models.cox_time.CoxTime):
    make_dataset = models.data.CoxVaccDataset

    def __init__(self, net, optimizer=None, device=None, shrink=0., labtrans=None, loss=None,
                 train_dict=None, val_dict=None):
        super().__init__(net, optimizer=optimizer, device=device, shrink=shrink, labtrans=labtrans, loss=loss,
                         sort_in_fit=False)
        self.train_dict = train_dict
        self.val_dict = val_dict

    @staticmethod
    def split_target_starts(target):
        # get starts from target if applicable
        durations, events, starts, vaccmap, is_val = target
        target = durations, events
        return target, starts, vaccmap, is_val

    def target_to_df(self, target):
        target, _, _, _ = self.split_target_starts(target)
        return super().target_to_df(target)

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
        input = [tt.tuplefy(i).iloc[idx_sort][0] for i in input]
        target = [tt.tuplefy(t).iloc[idx_sort][0] for t in target[:-1]] + [target[-1]]

        # split tuples to components
        input, time_var_input = input
        target, starts, vaccmap, is_val = self.split_target_starts(target)
        durations, events = target

        if is_val:
            saved_dict = self.val_dict
        else:
            saved_dict = self.train_dict

        dataset = self.make_dataset(input, time_var_input, durations, events, starts, vaccmap, n_control=n_control,
                                    cached_dict=saved_dict)

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