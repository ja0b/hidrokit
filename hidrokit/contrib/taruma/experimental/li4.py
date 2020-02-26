"""Experimental Module/Function. Based on Laporan Implementasi #4"""

import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import HydroErr as he
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hidrokit.contrib.taruma import hk43, hk53
from hidrokit.prep import timeseries

import seaborn as sns  # pylint: disable=import-error


def _check_system(PACKAGE_LIST='numpy pandas matplotlib'):
    from pkg_resources import get_distribution
    from sys import version_info

    print(':: INFORMASI VERSI SISTEM')
    print(':: {:>12s} version: {:<10s}'.format(
        'python',
        '{}.{}.{}'.format(*version_info[:3]))
    )
    for package in PACKAGE_LIST.split():
        print(':: {:>12s} version: {:<10s}'.format(
            package,
            get_distribution(package).version)
        )


def _check_tf():
    import tensorflow as tf  # pylint: disable=import-error
    print(':: {:>12s} version: {:<10s}'.format(
        'tensorflow',
        tf.__version__)
    )
    print(':: {:>12s} version: {:<10s}'.format(
        'keras',
        tf.keras.__version__)
    )


@contextmanager
def timeit_context(process, info_list):
    starttime = datetime.now(timezone(timedelta(hours=7)))
    str_start = starttime.strftime("%Y%m%d %H:%M")
    print(f':: {process} START: {str_start}')
    yield
    endtime = datetime.now(timezone(timedelta(hours=7)))
    str_end = endtime.strftime("%Y%m%d %H:%M")
    print(f':: {process} END: {str_end}')
    elapsedtime = endtime - starttime
    print(f':: {process} DURATION: {elapsedtime.seconds/60:.2f} min')
    info_list[process] = [str_start, str_end, elapsedtime.seconds / 60]


def clean_title(title):
    new = re.sub(r'\W+', ' ', title).lower()
    return new.replace(' ', '_')


def _parse_stations(stations):
    return stations.replace(' ', '').split(',')


def find_invalid(df):
    results = {}
    for col in df.columns:
        results[col] = hk43._check_invalid(df.loc[:, col].values)
    return results


def plot_corr_mat(df, savefig=False, _DIRIMAGE=None):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.tight_layout()
    sns.heatmap(
        corr, mask=mask, cmap='RdGy',
        center=0, square=True, robust=True,
        linewidth=.5, cbar_kws={'shrink': .7}, annot=True, ax=ax,
        fmt='.2f',
            annot_kws={'fontsize': 'large'})
    ax.set_title('Matriks Korelasi Dataset',
                 fontweight='bold', fontsize='xx-large')

    if savefig:
        plt.savefig(
            _DIRIMAGE / 'grafik_korelasi_matriks.png',
            dpi=150)

    return fig, ax


def plot_pairplot(df, savefig=False, _DIRIMAGE=None):
    grid = sns.pairplot(df, markers='+')
    fig = grid.fig
    fig.set_size_inches(15, 15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle('Grafik PairPlot Dataset',
                 fontweight='bold', fontsize='xx-large')

    if savefig:
        plt.savefig(
            _DIRIMAGE / 'grafik_pairplot_dataset.png',
            dpi=150)

    return grid


def train_test_split_ann(data, timesteps, date_start, target_column=None):
    feature_column = (
        data.columns[:-1] if target_column is None else
        data.drop(target_column, axis=1).columns
    )

    table_ts = timeseries.timestep_table(
        data, columns=feature_column, keep_first=False,
        timesteps=timesteps
    )

    train = table_ts.loc[slice(None, date_start)][:-1]
    test = table_ts.loc[slice(date_start, None)]

    return train, test


def train_test_split_rnn(data, timesteps, date_start,
                         feature_columns=None, target_column=None):

    feature_columns = (
        data.columns.to_list()[:-1] if feature_columns is None else
        feature_columns
    )

    target_column = (
        data.columns.to_list()[-1:] if target_column is None else
        target_column
    )

    rnn_X, rnn_y = hk53.tensor_array(
        data, X_columns=feature_columns,
        timesteps=timesteps,
        y_out=True, y_columns=target_column
    )

    ix_split = data.index.get_loc(date_start) - timesteps

    X_train = rnn_X[:ix_split, :, :]
    y_train = rnn_y[:ix_split]

    X_test = rnn_X[ix_split:, :, :]
    y_test = rnn_y[ix_split:]

    return (X_train, y_train), (X_test, y_test)


def save_history_to_csv(history, path):
    df = pd.DataFrame(history.history)
    df.to_csv(path)
    return df


def concat_train_test(model, X_train, y_train, X_test, y_test, df, ts, sc_y):
    dfindex = df[ts:].index

    sim_train = model.predict(X_train).flatten()
    sim_test = model.predict(X_test).flatten()
    sim = np.concatenate([sim_train, sim_test], axis=0)
    real_sim = sc_y.inverse_transform(sim)

    obs = np.concatenate([y_train, y_test], axis=0)
    real_obs = sc_y.inverse_transform(obs)

    result = pd.DataFrame(
        index=dfindex,
        data=np.stack([real_sim, real_obs], axis=1),
        columns=['SIMULATED', 'OBSERVED']
    )

    return result


def xy_split(data):
    return data.values[:, :-1], data.values[:, -1]


def _calc_r2_res(df, slice_model):
    sim = df[slice_model].SIMULATED.values
    obs = df[slice_model].OBSERVED.values
    return he.r_squared(sim, obs)


def plot_r2_deep(model_list, slice_model, ts=5, ts_col=0, title='TRAIN SET',
                 savefig=False, _DIRIMAGE=None):
    fig, ax = plt.subplots(
        nrows=1, ncols=4, figsize=(20, 5)
    )
    _TITLE = ['ANN', 'RNN', 'LSTM', 'GRU']

    for i, dl_model in enumerate(model_list):
        data = dl_model[ts_col]
        sns.regplot(
            x='SIMULATED', y='OBSERVED', data=data[slice_model],
            ax=ax[i], marker='o',
            scatter_kws={'s': 20, 'alpha': 0.7}
        )
        ax[i].set_title('{}'.format(
            _TITLE[i]), fontweight='bold')
        ax[i].text(0.05, 0.95, '$R^2={{{:.5f}}}$'.format(
            _calc_r2_res(data, slice_model)), transform=ax[i].transAxes,
            ha='left', va='top', fontsize='large',
            bbox=dict(boxstyle='square', fc='w', alpha=0.6, ec='k'))
        ax[i].grid(True, axis='both')

    fig.suptitle(
        f'Grafik $R^2$, {title}, $timesteps={{{ts}}}\\ \\mathit{{hari}}$',
        size='xx-large', fontweight='bold', y=0.99
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if savefig:
        plt.savefig(
            _DIRIMAGE / 'grafik_r2_{}_ts{}.png'.format(
                clean_title(title), ts
            ),
            dpi=150)

    return fig, ax
