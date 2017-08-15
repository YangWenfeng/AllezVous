"""
Use DBSCAN clustering latitude & longitude on properties
Inspire by http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
"""
import time
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
# from sklearn import metrics
from geopy.distance import great_circle
from shapely.geometry import MultiPoint


# read properties data
print 'Read properties data.'
properties = pd.read_csv("../data/properties_2016.csv")
# properties = pd.read_csv("../data_debug/properties_2016.csv")
train_coordinates = properties[['latitude', 'longitude']]
train_coordinates = train_coordinates.astype(np.float32)
train_coordinates = train_coordinates.fillna(train_coordinates.mean(), inplace=True)

coordinates = train_coordinates.as_matrix(columns=['latitude', 'longitude']) / 1e6

print 'Run DBSCAN cluster.'
start_time = time.time()
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
            metric='haversine').fit(np.radians(coordinates))

cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
print('Number of clusters: {}'.format(num_clusters))

# all done, print the outcome
message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(coordinates), num_clusters, 100*(1 - float(num_clusters) / len(coordinates)),
                     time.time()-start_time))
# take long time
# print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(coordinates, cluster_labels)))

clusters = pd.Series([coordinates[cluster_labels == n] for n in xrange(num_clusters)])
print clusters.head()

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

print 'Find the point in each cluster that is closest to its centroid.'
centermost_points = clusters.map(get_centermost_point)

# unzip the list of centermost points (latitude, longitude) tuples into separate latitude and longitude lists
latitude, longitude = zip(*centermost_points)

# from these latitude/longitude create a new df of one representative point for each cluster
rep_coordinates = pd.DataFrame({'longitude': longitude, 'latitude': latitude})
print rep_coordinates.head()

properties_latlng_cluster = properties['parcelid']
properties_latlng_cluster['cluster_label'] = cluster_labels
properties_latlng_cluster['cluster_latitude'] = [latitude[label] for label in cluster_labels]
properties_latlng_cluster['cluster_longitude'] = [longitude[label] for label in cluster_labels]

properties_latlng_cluster.to_csv('../data/properties_latlng_cluster.csv', index=False)

print 'done'
