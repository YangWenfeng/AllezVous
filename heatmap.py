import numpy as np
import pandas as pd
import json

train = pd.read_csv("../data/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("../data/properties_2016.csv")
train_with_properties = train.merge(properties, on='parcelid', how='left')
# File heatmap/data.js is like:
"""
var data = {
    "select_option_name1": {"max": max_value, "data": [{"lat": x, "lng": y, "cnt": n}, ...]},
    "select_option_name2": {"max": max_value, "data": [{"lat": x, "lng": y, "cnt": n}, ...]}, ...
};
"""

data = {}
groupby_columns = ['regionidcity', 'regionidneighborhood', 'regionidzip']
for df, name in zip([train_with_properties, properties], ['train', 'prop']):
    for feature in groupby_columns:
        # 1e6
        lat_dict = df.groupby([feature])['latitude'].mean()/1e6
        lng_dict = df.groupby([feature])['longitude'].mean()/1e6
        cnt_dict = df.groupby([feature])['parcelid'].count()

        tmp = pd.DataFrame({'lat': lat_dict.values, 'lng': lng_dict.values, 'cnt': cnt_dict.values})
        js = tmp.to_json(orient='records')
        key = '%s_%s_json' % (name, feature)
        val = {'max': cnt_dict.values.max(), 'data': json.loads(js)}
        data[key] = val

# outlier
# The lower and upper bounds are gotten from github.com/andrewpiggy
OUTLIER_UPPER_BOUND = 0.419
OUTLIER_LOWER_BOUND = -0.4

outlier_index = []
for i in xrange(len(train_with_properties)):
    if train_with_properties['logerror'][i] >= OUTLIER_UPPER_BOUND or\
                    train_with_properties['logerror'][i] <= OUTLIER_LOWER_BOUND:
        outlier_index.append(i)


train_with_properties_outlier = train_with_properties.iloc[outlier_index]

for feature in groupby_columns:
    # 1e6
    lat_dict = train_with_properties_outlier.groupby([feature])['latitude'].mean()/1e6
    lng_dict = train_with_properties_outlier.groupby([feature])['longitude'].mean()/1e6
    cnt_dict = train_with_properties_outlier.groupby([feature])['parcelid'].count()

    tmp = pd.DataFrame({'lat': lat_dict.values, 'lng': lng_dict.values, 'cnt': cnt_dict.values})
    js = tmp.to_json(orient='records')
    key = 'outlier_%s_json' % feature
    val = {'max': cnt_dict.values.max(), 'data': json.loads(js)}
    data[key] = val

    train_dict = train_with_properties.groupby([feature])['parcelid'].count()
    # outlier / train for each region, scale = 1000
    ratio_dict = ((cnt_dict / train_dict)[cnt_dict.index].fillna(0) * 1000).astype(int)

    tmp = pd.DataFrame({'lat': lat_dict.values, 'lng': lng_dict.values, 'cnt': ratio_dict.values})
    js = tmp.to_json(orient='records')
    key = 'outlier_ratio_%s_json' % feature
    val = {'max': ratio_dict.values.max(), 'data': json.loads(js)}
    data[key] = val
f = open('heatmap/data.js', 'w')
f.write('var data = %s;' % json.dumps(data))
f.close()
