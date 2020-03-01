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


def target_table(
        df, columns=None, steps=[1, 5, 10],
        keep_first=False, format='{col}_plus{step}', shift=-1):

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


def loc_in_int(df, slice_data, PERIOD_SLICE=None):
    if PERIOD_SLICE is None:
        return (df.assign(no=np.arange(df.shape[0]))
                .loc[slice_data, 'no'].values)
    else:
        new_df = df.assign(no=np.arange(df.shape[0]))
        loc_period = df.index.get_loc(PERIOD_SLICE)
        loc_slice = new_df.loc[slice_data, 'no'].values
        return (
            loc_slice[loc_slice < loc_period],
            loc_slice[loc_slice >= loc_period] - loc_period
        )


def scaling_data(dataset, period_slice, ObjectStandardScaler):
    dataset_copy = dataset.copy()

    train = dataset_copy.loc[:period_slice][:-1]
    test = dataset_copy.loc[period_slice:]

    train_scale = train.copy()
    train_scale[:] = ObjectStandardScaler.fit_transform(train_scale[:])

    test_scale = test.copy()
    test_scale[:] = ObjectStandardScaler.transform(test_scale[:])

    return pd.concat([train_scale, test_scale], axis=0)


def ann_train_test_split(data, timesteps, date_start, feature_columns=None):
    from hidrokit.prep import timeseries
    feature_columns = (
        data.columns if feature_columns is None else feature_columns
    )

    table_ts = timeseries.timestep_table(
        data, columns=feature_columns, keep_first=True,
        timesteps=timesteps
    )

    train = table_ts.loc[slice(None, date_start)][:-1]
    test = table_ts.loc[slice(date_start, None)]

    return train, test


def save_scaler_attribute(ObjectStandardScaler, filepath, columns=None):
    size = ObjectStandardScaler.scale_.shape[0]
    data = dict(
        scale=ObjectStandardScaler.scale_,
        mean=ObjectStandardScaler.mean_,
        var=ObjectStandardScaler.var_,
        n_sample_seen=[ObjectStandardScaler.n_samples_seen_] * size
    )
    df = pd.DataFrame(data=data, index=columns).T
    df.to_csv(filepath)

    return df


def load_scaler_attribute(ObjectStandardScaler, filepath, column):
    df = pd.read_csv(filepath, index_col=0)
    select_df = df[column]

    for index, value in select_df.iteritems():
        setattr(ObjectStandardScaler, index + '_', value)

    return df


def transfer_scaler_attribute(InputScaler, TargetScaler, index_col=None):
    list_attribute = 'scale_ mean_ var_'.split()

    for attr in list_attribute:
        attr_val = getattr(InputScaler, attr)
        attr_val = (
            attr_val if index_col is None
            else attr_val[index_col]
        )
        setattr(TargetScaler, attr, attr_val)

    setattr(TargetScaler, 'n_samples_seen_',
            getattr(InputScaler, 'n_samples_seen_'))


def check_ann_rnn_array(
        ann_array, rnn_array, counter=5, verbose=False, kind='target'):

    val = 0
    for _ in range(counter):
        _CHECK = np.random.randint(0, ann_array.shape[0])
        if kind == 'target':
            cond = (rnn_array[_CHECK].flatten() ==
                    ann_array.iloc[_CHECK].values).all()
        elif kind == 'feature':
            cond = (rnn_array[_CHECK].T[:, ::-1].flatten()
                    == ann_array.iloc[_CHECK].values).all()
        val += cond
        if verbose:
            print(_CHECK, cond)

    if val == counter:
        print(':: ANN DAN RNN IDENTIK/SAMA')
