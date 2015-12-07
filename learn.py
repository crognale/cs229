import facedb
import eigen
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.spatial import distance
import scipy

#nearest neighbor index of weight vector to weight matrix

def closest_euclidean(pt, W, k):
	min_is = [-1] * k
	min_dists = [float('inf')] * k
	for i in range(W.shape[1]):
		dist = distance.euclidean(pt, W[:, i])

		max_i = max(xrange(len(min_dists)), key=min_dists.__getitem__)
		if dist < min_dists[max_i]:
			min_dists[max_i]= dist
			min_is[max_i] = i
	return min_is


def closest_mahalanobis(pt, W, cov, k):
	min_is = [-1] * k
	min_dists = [float('inf')] * k
	for i in range(W.shape[1]):
		dist = distance.mahalanobis(pt, W[:, i], cov)

		max_i = max(xrange(len(min_dists)), key=min_dists.__getitem__)
		if dist < min_dists[max_i]:
			min_dists[max_i]= dist
			min_is[max_i] = i
	return min_is

facedir = '10kfaces/'
fdb = facedb.FaceDB(facedir, 'test.db')
eig = eigen.Eigen(facedir, 1000, False)

X = []

k = 3000 # number of ratings to choose, low number for fast testing
ratings = fdb.ratings_for_user('sam')[0:k]

paths = []
for r in ratings:
	paths.append(facedir + r[0])

ignoreFirst = 10 #ignore the first x eigenfaces
weight_k = 50 #use this many eigenfaces total
weights = eig.weights_for_imgs(paths, ignoreFirst, weight_k)

for i in xrange(len(ratings)):
	#X is array of 3-tuples (filename, rating, weights)
	X.append((ratings[i][0], ratings[i][1], weights[:,i]))

#split X into training set (X) and testing set(Y)
split_ind = 1000
Y = X[split_ind:]
Y_weights = weights[:, split_ind:]

X = X[:split_ind]
X_weights = weights[:,:split_ind]



#testing validation (needs work)
cov = np.linalg.inv(np.diag(eig.S[ignoreFirst:ignoreFirst + weight_k]))
for test_i in range(len(Y)):
	test_w = Y_weights[:,test_i].reshape(-1, 1)
	#look only at positive test examples for now
	if Y[test_i][1] == 1:
		#find k-nearest neighbors using euclidian and mahalanobis
		i_eucs = closest_euclidean(test_w, X_weights, 5)
		i_mahs = closest_mahalanobis(test_w, X_weights, cov, 5)
		#print out k-nearest neighbor ratings of the test example
		print Y[test_i][1], '==>', [X[i_mah][1] for i_mah in i_mahs]

#create average attractive face
avg_w = np.zeros((eig.U.shape[1], 1))
pos_count = 0
for i in range(len(X)):
	if X[i][1] == 1:
		avg_w[ignoreFirst:ignoreFirst+weight_k,:] += X[i][2].reshape(-1, 1)
		pos_count += 1
avg_w /= pos_count
recon = eig.mu.reshape(-1,1) + np.dot(eig.U, avg_w)
scipy.misc.imsave('pos_recon.jpg', recon.reshape(256, 256))
