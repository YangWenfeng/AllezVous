"""
Use K-means clustering latitude & longitude on properties
"""
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist


# read properties data
print 'Read properties data.'
prop_df = pd.read_csv("input/properties_2016.csv")
X = prop_df[['latitude', 'longitude']]
X = X.fillna(X.mean(), inplace=True)
X = X.astype(np.float32)

print 'K-means cluster'
K = range(25, 501, 25)
KM = [kmeans(X, k) for k in K]
# pickle.dump(KM, open('output/km_cluster_latlng.pl', 'wb'))

print 'Calculate elbow curve'
centroids = [cent for (cent, var) in KM]
D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d)/X.shape[0] for d in dist]

print 'Save K/KM/D_k/avgWithinSS to pickle file '
result = {
    'K': K,
    'KM': KM,
    'D_k': D_k,
    'avgWithinSS': avgWithinSS
}
pickle.dump(KM, open('output/km_cluster_latlng_elbow.pl', 'wb'))

print 'done'
