"""
AllezVous 1.0, by Yang
1. base zestimate_algorithm_main.py
2. add logerror tag as feature
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import gc


def reduce_categs(props):
    def reduce_one_categs(df, col, threshold, new_val):
        value_counts = df[col].value_counts()
        to_remove = value_counts[value_counts <= threshold].index
        df[col].replace(to_remove, new_val, inplace=True)
        return df

    props = reduce_one_categs(props, 'regionidcity', 5000, 0)
    props = reduce_one_categs(props, 'regionidneighborhood', 3000, 0)
    props = reduce_one_categs(props, 'regionidzip', 3000, 0)


def fill_missing(props):
    props['taxdelinquencyflag'].fillna('N', inplace=True)
    props['airconditioningtypeid'].fillna(5, inplace=True)
    props['architecturalstyletypeid'].fillna(7, inplace=True)
    props['basementsqft'].fillna(0, inplace=True)
    props['bathroomcnt'].fillna(0, inplace=True)
    props['bedroomcnt'].fillna(1, inplace=True)
    props['buildingclasstypeid'].fillna(4, inplace=True)
    props['buildingqualitytypeid'].fillna(7, inplace=True)
    props['decktypeid'].fillna(0, inplace=True)
    props['poolcnt'].fillna(0, inplace=True)
    props['poolsizesum'].fillna(0, inplace=True)
    props['pooltypeid10'].fillna(0, inplace=True)
    props['pooltypeid2'].fillna(0, inplace=True)
    props['pooltypeid7'].fillna(0, inplace=True)
    props['regionidcity'].fillna(0, inplace=True)
    props['regionidcounty'].fillna(0, inplace=True)
    props['regionidneighborhood'].fillna(0, inplace=True)
    props['regionidzip'].fillna(0, inplace=True)
    props['taxdelinquencyyear'].fillna(0, inplace=True)

    drop_cols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                 'propertycountylandusecode', 'latitude', 'longitude']
    for c, dtype in zip(props.columns, props.dtypes):
        if c in drop_cols:
            continue
        if props[c].isnull().values.any():
            props[c].fillna(props[c].median(), inplace=True)



def bind_float32(props):
    for c, dtype in zip(props.columns, props.dtypes):
        if dtype == np.float64:
            props[c] = props[c].astype(np.float32)

def bind_obj(props):
    for c in props.dtypes[props.dtypes == object].index.values:
        print 'features of Object type', c
        if c == 'taxdelinquencyflag':
            props[c] = (props[c] == 'Y')
        else:
            props[c] = (props[c] == True)


def one_hot_encoding(df, cols):
    for col in cols:
        # Perform encoding on training data
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, axis=1)
        df = df.join(one_hot)


def get_logerror_tags(train):
    def parse_logerror_tag(v):
        left_split, right_split = -0.35, 0.45
        if v <= left_split:
            return 0
        elif v >= right_split:
            return 1
        else:
            return 2
    return np.array([parse_logerror_tag(v) for v in train['logerror']])

def logerror_tag_classfication(train, props, sample):
    print 'Merge train and properties'
    df_train = train.merge(props, how='left', on='parcelid')

    print 'Build x_train, y_train'
    drop_cols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                 'propertycountylandusecode', 'latitude', 'longitude']

    x_train = df_train.drop(drop_cols, axis=1)
    y_train = get_logerror_tags(train)
    train_cols = x_train.columns

    del df_train; gc.collect()

    split = 80000

    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

    print ('Building DMatrix...')

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    del x_train, x_valid; gc.collect()

    print ('Training...')

    # use softmax multi-class classification
    params = {'eta': 0.02, 'objective': 'multi:softmax', 'max_depth': 6, 'silent': 1, 'nthread': 4, 'num_class': 3}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

    del d_train, d_valid; gc.collect()

    print ('Building test set...')

    sample['parcelid'] = sample['ParcelId']

    df_test = sample.merge(props, how='left', on='parcelid')
    x_test = df_test[train_cols]
    print ('Predicting on test...')
    d_test = xgb.DMatrix(x_test)
    p_test = clf.predict(d_test)

    logerror_tag_df = pd.DataFrame({'ParcelId': sample['ParcelId'], 'logerror_tag': p_test})
    logerror_tag_df.to_csv('output/logerr_tag_classfication.csv', index=False, float_format='%.4f')

    print 'logerror_tag_classfication done'
    return logerror_tag_df

def logerror_regression(train, props, sample):
    drop_cols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                 'propertycountylandusecode', 'latitude', 'longitude']
    df_train = train.merge(props, how='left', on='parcelid')
    x_train = df_train.drop(drop_cols, axis=1)
    y_train = df_train['logerror'].values
    train_cols = x_train.columns

    split = 80000

    x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

    print ('Building DMatrix...')

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    del x_train, x_valid; gc.collect()

    print ('Training...')

    params = {'eta': 0.02, 'objective': 'reg:linear', 'eval_metric': 'mae', 'max_depth': 4, 'silent': 1}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

    del d_train, d_valid; gc.collect()

    print ('Building test set...')
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(props, how='left', on='parcelid')
    x_test = df_test[train_cols]
    d_test = xgb.DMatrix(x_test)

    del x_test; gc.collect()

    print ('Predicting on test...')

    p_test = clf.predict(d_test)
    sub_df = sample.copy()
    sub_df = sub_df.drop(['parcelid'], axis=1)

    for c in sub_df.columns[sub_df.columns != 'ParcelId']:
        sub_df[c] = p_test
    print ('Writing to CSV...')
    sub_df.to_csv('output/sample_submission_allezvous.v1.csv', index=False, float_format='%.4f')

    return sub_df

print 'Loading data...'
train = pd.read_csv('input/train_2016_v2.csv')
props = pd.read_csv('input/properties_2016.csv')
sample = pd.read_csv('input/sample_submission.csv')

print 'Processing properties...'
reduce_categs(props)
fill_missing(props)
bind_float32(props)
bind_obj(props)

one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty',
                       'regionidneighborhood',
                       'regionidzip']
one_hot_encoding(props, one_hot_encode_cols)

logerror_tag_df = logerror_tag_classfication(train, props, sample)
print logerror_tag_df.head()
#
# logerror_tag_df = pd.read_csv('output/logerr_tag_classfication.csv')
props['logerror_tag'] = logerror_tag_df['logerror_tag']
one_hot_encoding(props, ['logerror_tag'])

sub_df = logerror_regression(train, props, sample)
print sub_df.head()

