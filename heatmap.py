import pandas as pd

train_df = pd.read_csv("input/train_2016_v2.csv", parse_dates=["transactiondate"])
prop_df = pd.read_csv("input/properties_2016.csv")
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

data = []
for df, name in zip([train_df, prop_df], ['train', 'prop']):
    for feature in ['regionidcounty', 'regionidcity', 'regionidneighborhood', 'regionidzip']:
        # 1e6
        lat_dict = df.groupby([feature])['latitude'].mean()/1e6
        lng_dict = df.groupby([feature])['longitude'].mean()/1e6
        cnt_dict = df.groupby([feature])['parcelid'].count()

        nbh_pdf = pd.DataFrame({'lat': lat_dict.values, 'lng': lng_dict.values, 'cnt': cnt_dict.values})
        nbh_json = nbh_pdf.to_json(orient='records')

        data.append('var %s_%s_json = {max: %d, data: %s};' % (name, feature, cnt_dict.values.max(), nbh_json))

f = open('heatmap/data.js', 'w')
f.write('\n'.join(data))
f.close()