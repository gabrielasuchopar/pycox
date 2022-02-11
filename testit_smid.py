import pandas as pd

from pycox.models.cox_vacc import CoxVacc

y = ['duration', 'event']

import glob

path = '/mnt/c/Users/gabis/Documents/vyvanuti/smid_process.csv'
print(path)
df = pd.read_csv(path, nrows=500000)
df
print('read')

df = df[['T1', 'T2', 'Infected', ' InfPrior', 'VaccStatus', ' AgeGr', ' Sex ']].copy()
df.rename(columns={'T2': 'duration', 'Infected': 'event'}, inplace=True)
df_splitted = df

df_splitted = df
to_onehot = [' InfPrior', 'VaccStatus', ' AgeGr', ' Sex ']


def manual_onehot(data, col):
    if data[col].isna().any():
        data[f'{col}_nan'] = data[col].isna()

    uniques = data[col].unique()
    print(uniques)

    if len(uniques) == 2:
        data[f'{col}_onehot'] = (data[col] == uniques[0]).astype(int)
    else:
        for val in uniques:
            print(val)
            data[f'{col}_{val}'] = (data[col] == val).astype(int)


for c in to_onehot:
    print(c)
    manual_onehot(df_splitted, c)


X_y = df_splitted
X_y.drop(columns=to_onehot, inplace=True)
X_y

import numpy as np
np.random.seed(42)

X_y_copy = X_y#.copy()#.iloc[:32000]
#X_y_copy.drop(columns=['index'], inplace=True)

use_sample = False

if use_sample:
    size = 32000
    xyidx = np.random.choice(X_y_copy.index, replace=False, size=size)
    X_y_copy = X_y_copy.iloc[xyidx]


def sample_fast(data, sample_size=0.2, random_state=42):
    test = data.sample(frac=0.2, random_state=random_state)
    print('Test Done')

    mmmap = ~data.index.isin(test.index)
    return test, data[mmmap]

df_test, df_train = sample_fast(X_y_copy)
print(len(df_train))
df_val, df_train = sample_fast(df_train)

x_train = df_train.astype('float32').drop(columns=y).to_numpy()
x_val = df_val.astype('float32').drop(columns=y).to_numpy()
x_test = df_test.astype('float32').drop(columns=y).to_numpy()

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.preprocessing.label_transforms import LabTransCoxTime
import torchtuples as tt

labtrans = LabTransCoxTime()
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)

y_train = get_target(df_train)
y_val = get_target(df_val)

import torch
import torchtuples as tt

in_features = x_train.shape[1]
num_nodes = [32, 32, 64, 128]
out_features = 1
batch_norm = False
dropout = 0.1
output_bias = False

#net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
#                              dropout, output_bias=output_bias)

net = MLPVanillaCoxTime(in_features + 2, num_nodes, batch_norm, dropout)
#net = Net(in_features, num_nodes, out_features, batch_norm,
#          dropout, output_bias=output_bias)

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

model = CoxVacc(net, tt.optim.Adam, labtrans=labtrans) #device=torch.device('cuda'))
model.optimizer.set_lr(0.0001)

epochs = 100
callbacks = [#tt.callbacks.ClipGradNorm(net, 6), #tt.callbacks.LRScheduler(),
# tt.callbacks.EarlyStopping()
]
verbose = True

batch_size=1024
ones_train = (df_train['event'] == 0) | (df_train['event'] == 1)
ones_val = (df_val['event'] == 0) | (df_val['event'] == 1)

starts = df_train['T1'].to_numpy()
#starts, _ = labtrans.transform(starts, np.zeros((0,)))

valstarts = df_val['T1'].to_numpy()
#valstarts, _ = labtrans.transform(valstarts, np.zeros((0,)))

#vaccmap_train = ones_train
#vaccmap_val = ones_val
vaccmap_train = df_train['VaccStatus__unvacc'] == 0
vaccmap_val = df_val['VaccStatus__unvacc'] == 0

y_train = (*y_train, starts, vaccmap_train.reset_index(drop=True), False)
y_val = (*y_val, valstarts, vaccmap_val.reset_index(drop=True), True)

#x_val = (x_val, np.stack([starts, starts], axis=1))
#x_train = (x_train, np.stack([starts, starts], axis=1))
val = tt.tuplefy(x_val, y_val)
dummy_var = np.stack([starts, starts], axis=1)
dummy_var_val = np.stack([valstarts, valstarts], axis=1)

batch_size = 2048
epochs = 5
log = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                val_data=val, val_batch_size=batch_size, shuffle=True, n_control=100,
                time_var_input=dummy_var, val_time_var_input=dummy_var_val)


# predicts hazard for a subject in time T since vaccination
def pred_hazard_ratio(T, starts, vaccmap, real_time=False):
    # for vaccinated, the input time is time T (e.g. 30 days since 2nd dose)
    valdur = np.zeros((len(x_train),), dtype=np.float32) + T

    val_pred = tt.tuplefy(x_train,
                          valdur)  # for prediction, we need to tuplefy it with X (due to ugly design of pycox :( )

    # predict the hazard
    pred_val = model.predict(val_pred, starts, vaccmap, time_is_real_time=real_time, time_var_input=dummy_var)

    return pred_val

r1 = pred_hazard_ratio(5, starts, vaccmap_train)
r2 = pred_hazard_ratio(300, starts, vaccmap_train, real_time=True)

print(r1)
print(r2)

model.compute_baseline_hazards(verbose=True)

print()
