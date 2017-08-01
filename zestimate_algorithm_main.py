"""
https://github.com/GarrettGenz/Zestimate-Algorithm/blob/master/main.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import matplotlib.pyplot as plt

def showPlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=10)
    plt.show()


def one_hot_encoding(cols, train):
    for col in cols:
        # Perform encoding on training data
        one_hot = pd.get_dummies(train[col], prefix=col)
        if col <> "playerid":
            train = train.drop(col, axis=1)
        train = train.join(one_hot)

    return train

def reduce_categs(df, col, threshold, new_val):
    value_counts = df[col].value_counts()
    to_remove = value_counts[value_counts <= threshold].index
    df[col].replace(to_remove, new_val, inplace=True)

    return df


print ('Loading data...')

train = pd.read_csv('input/train_2016_v2.csv')
props = pd.read_csv('input/properties_2016.csv')
sample = pd.read_csv('input/sample_submission.csv')

#print props['regionidcity'].value_counts()
#print props['regionidcounty'].value_counts()
#print props['regionidneighborhood'].value_counts()
#print props['regionidzip'].value_counts()

props = reduce_categs(props, 'regionidcity', 5000, 0)
props = reduce_categs(props, 'regionidneighborhood', 3000, 0)
props = reduce_categs(props, 'regionidzip', 3000, 0)

print ('Binding to float32...')

drop_cols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'latitude', 'longitude']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip']
print props['pooltypeid7'].unique()

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

for c, dtype in zip(props.columns, props.dtypes):
    if c not in drop_cols:
        if dtype == np.float32:
            props[c] = props[c].astype(np.float32)
    #    print c, dtype
    #    s = props[c]
 #       print props[c].describe()
        if props[c].isnull().values.any():
            props[c].fillna(props[c].median(), inplace=True)
 #       print props[c].head()
    #    print s.describe()



print ('One hot encode categorical columns...')

props = one_hot_encoding(one_hot_encode_cols, props)

print ('Create training set...')

df_train = train.merge(props, how='left', on='parcelid')

x_train = df_train.drop(drop_cols, axis=1)
y_train = df_train['logerror'].values

#print (x_train)
#print (x_train.shape, y_train.shape)

train_cols = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    print c
    if c == 'taxdelinquencyflag':
        x_train[c] = (x_train[c] == 'Y')
    else:
        x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

#lasso_x_train = x_train
#lasso_y_train = y_train

split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print ('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

#print d_train.feature_types

del x_train, x_valid; gc.collect()

print ('Training...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid; gc.collect()

print ('Building test set...')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(props, how='left', on='parcelid')

del props; gc.collect()

x_test = df_test[train_cols]

print ('Trouble loop...')
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    print c
    if c == 'taxdelinquencyflag':
        x_test[c] = (x_test[c] == 'Y')
    else:
        x_test[c] = (x_test[c] == True)
print ('End trouble loop...')

del df_test, sample; gc.collect()

print ('Problem')
d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print ('Predicting on test...')

p_test = clf.predict(d_test)

# print ('Predicting on test with Lasso...')
#
# from sklearn.linear_model import Lasso
#
# best_alpha = .001
#
# regr = Lasso(alpha=best_alpha, max_iter=50000)
# regr.fit(lasso_x_train, lasso_y_train)
#
# p_lasso = regr.predict(d_test)
#
# del d_test; gc.collect()
#
# ##################################################
#
# p_combined = (p_test + p_lasso) / 2

sub = pd.read_csv('data/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print ('Writing to CSV...')
sub.to_csv('output/zestimate_algorithm_main.csv', index=False, float_format='%.4f')

print (sub.shape)
print (sub.iloc[407820])
print (sub.iloc[407821])
print (sub.iloc[407822])