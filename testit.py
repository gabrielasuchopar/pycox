import pandas as pd

y = ['duration', 'event']

import glob

df = None

for path in glob.glob('splitted_data_*.csv'):
    print(path)
    next_df = pd.read_csv(path)

    df = next_df if df is None else pd.concat([df, next_df], ignore_index=True)

# proper time 0
vacc_date = pd.to_datetime("2020-12-26")
# vacc_date = pd.to_datetime(data_fit['prvnidavka'])
max_date = pd.to_datetime("2021-11-20")

for c in ['death']:
    if c != 'death':
        col_date = df[c]
    else:
        col_date = pd.to_datetime(df[c])

    print(c, col_date.min(), col_date.max())

    # drop data errors
    big_ids = df[col_date > pd.to_datetime("today")].index
    small_ids = df[col_date < pd.to_datetime("2020-01-01")].index
    df.drop(index=big_ids, inplace=True)
    df.drop(index=small_ids, inplace=True)

    # time since time 0
    t_since_first = (col_date - vacc_date).dt.days
    df[f"{c}_int"] = t_since_first

    df[f"{c}_flag"] = (~t_since_first.isna()).astype(int)

    # fill NaTs with max, flag is 0
    dmax = (max_date - vacc_date).days
    print(dmax)

    df[f"{c}_int"].fillna(dmax, inplace=True)

deadmap = ~df['death'].isna() & (df['duration'] == 329.0)
df.loc[deadmap, 'duration'] = df.loc[deadmap, 'death_int']
df.loc[deadmap, 'event'] = 0

wrong_durations = df['duration'] < 0
df.loc[wrong_durations, 'duration'] = 329.0
df.loc[wrong_durations, 'event'] = 0

# TODO remove this as soon as you fix the index
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop(columns=[c for c in df.columns if 'death' in c], inplace=True)
df.head()

df_splitted = df

to_onehot = ['pohlavi', 'OckovaciLatka1', 'OckovaciLatka2', 'OckovaciLatka3']


def manual_onehot(data, col):
    if data[col].isna().any():
        data[f'{col}_nan'] = data[col].isna()

    uniques = data[col].unique()
    print(uniques)
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
val = tt.tuplefy(x_val, y_val)

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

net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
#net = Net(in_features, num_nodes, out_features, batch_norm,
#          dropout, output_bias=output_bias)

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

model = CoxTime(net, tt.optim.Adam, labtrans=labtrans) #device=torch.device('cuda'))
model.optimizer.set_lr(0.0001)

epochs = 100
callbacks = [#tt.callbacks.ClipGradNorm(net, 6), #tt.callbacks.LRScheduler(),
# tt.callbacks.EarlyStopping()
]
verbose = True

batch_size=1024

starts = df_train['T'].to_numpy()
starts, _ = labtrans.transform(starts, np.zeros((0,)))
valstarts, _ = labtrans.transform(df_val['T'].to_numpy(), np.zeros((0,)))
starts

durs = df_train['duration'].to_numpy()
durs, _ = labtrans.transform(durs, np.zeros((0,)))

durs_i = np.argsort(durs)
st_i = np.argsort(starts)


def get_masked_durs(in_df, zero_nonvacc=True):
    durs = in_df['duration'].copy()

    # time zero for nonvacc
    if zero_nonvacc:
        durs.loc[(in_df['T'] == 0) & (in_df['prvnidavka_flag'] == 0)] = 0
    durs.loc[(in_df['T'] > 0)] = durs - in_df['T']

    return durs

#train_durs = get_masked_durs(df_train)
#val_durs = get_masked_durs(df_val)

vacc_train = (df_train['prvnidavka_flag'] == 1) | (df_train['preinfection_int'] > 0)
vacc_val = (df_val['prvnidavka_flag'] == 1) | (df_val['preinfection_int'] > 0)

#y_train = (*y_train, starts)
#y_val = (*y_val, valstarts)
y_train = (*y_train, starts, vacc_train)
y_val = (*y_val, valstarts, vacc_val)
val = tt.tuplefy(x_val, y_val)

batch_size = 2048
epochs = 20
log = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                val_data=val, val_batch_size=batch_size, shuffle=True, n_control=100,
                use_starts=True)

print()
