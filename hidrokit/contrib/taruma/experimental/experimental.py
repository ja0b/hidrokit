import pandas as pd
import numpy as np


def check_dates(datetime_series, freq='H'):
    range = datetime_series.iloc[0], datetime_series.iloc[-1]
    diff = pd.date_range(*range, freq=freq).difference(datetime_series)

    if diff.size == 0:
        print(':: DATA LENGKAP')
    else:
        print(':: DATA TIDAK LENGKAP')
        return diff


def target_table(df, columns=None, steps=[1, 5, 10], keep_first=False, format='{col}_plus{step}', shift=-1):

    columns = df.columns if columns is None else columns
    res_df = df[columns].copy()

    for col in columns:
        for step in steps:
            col_name = format.format(col=col, step=step)
            res_df[col_name] = res_df[col].shift(periods=shift * step)

    if keep_first:
        return res_df
    else:
        return res_df.drop(columns, axis=1)


def loc_in_int(df, slice_data):
    return df.assign(no=np.arange(df.shape[0])).loc[slice_data, 'no'].values
