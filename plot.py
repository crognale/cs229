import eigen
import facedb
import numpy as np
from os.path import isfile
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import scipy.misc

np.set_printoptions(threshold='nan')


facedir = '10kfaces/'
username = 'sam'
num_generated_eigenfaces = 1000
m = 3000 #max number of ratings to use

ignoreFirst = 5 #ignore the first x eigenfaces
weight_k = 400 #use this many eigenfaces total

fdb = facedb.FaceDB(facedir, 'test.db')
ratings = fdb.ratings_for_user(username)[0:m]
paths = [facedir + r[0] for r in ratings]

eig = eigen.Eigen(facedir, num_generated_eigenfaces, False)
print 'getting weights:'
weights = eig.weights_for_imgs(paths, ignoreFirst, weight_k).T
print 'weights: ',weights.shape

#labels = np.array([r[1] for r in ratings]).reshape(-1, 1)
labels = np.ravel([r[1] for r in ratings])



X = StandardScaler().fit_transform(weights)
y = labels
print 'X: ', X.shape
print 'y: ', y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print'percent no in training: ', 1- (float(y_train.sum()) / y_train.shape[0])

'''
clf = GaussianNB()
#clf = LinearDiscriminantAnalysis()
#clf = KNeighborsClassifier(1)
#clf = SVC()
#clf = QuadraticDiscriminantAnalysis()

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print 'overall score: ', score
'''

X_pos = np.array([X_train[i,:] for i in range(X_train.shape[0])
		if y_train[i] == 1])
y_pos = np.ravel([y for y in y_train if y == 1])
print X_pos[0,:]

params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(X_pos)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

kde = grid.best_estimator_

#draw from distribution
samples = kde.sample(10)
for i in range(samples.shape[0]):
	sample_recon =  np.dot(samples[i,:], eig.U[:,ignoreFirst:ignoreFirst+weight_k].T)
	sample_recon_mu =  (eig.mu / 255) + sample_recon
	scipy.misc.imsave('sample_{}.jpg'.format(i), sample_recon.reshape(256,256))
	scipy.misc.imsave('sample_mu{}.jpg'.format(i), sample_recon_mu.reshape(256,256))




scores = kde.score_samples(X_test)
avg_0 = 0.0
count_0 = 0.0
avg_1 = 0.0
count_1 = 0.0
for i in range (scores.shape[0]):
	#print scores[i], y_test[i]
	if y_test[i] == 0:
		avg_0 += scores[i]
		count_0 += 1
	else:
		avg_1 += scores[i]
		count_1 += 1
avg_0 = avg_0 / count_0
avg_1 = avg_1 / count_1
print 'avg0: ', avg_0, 'avg1: ', avg_1


'''
plt.scatter(X_test[:,10], X_test[:,11], c=y_test)
plt.show()
'''


'''
pos_score = clf.score(X_pos, y_pos)
print y_pos.shape
print 'pos score: ', pos_score
'''
