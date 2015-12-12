from eigen import open_preprocess
import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KernelDensity
import scipy
import facedb
from sklearn.naive_bayes import GaussianNB

import os
import numpy as np


n_row, n_col = 10,10
n_components = n_row * n_col
image_shape = (256,256)
rng = RandomState(0)

pca_n = 1000

img_dir_path = '10kfaces/'
files = os.listdir(img_dir_path)

faces = np.zeros((pca_n, image_shape[0] * image_shape[1]))
cur_n = 0
print 'reading images'
for f in files[0:pca_n]:
  target_file = img_dir_path + str(f)
  faces[cur_n] = open_preprocess(target_file)
  cur_n += 1

n_samples, n_features = faces.shape

#global centering
faces_centered = faces - faces.mean(0)
#local centering
#faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

###############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
     False)#,
]

###############################################################################
# Plot a sample of the input data

#plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it

'''
for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    else:
        components_ = estimator.components_
    if hasattr(estimator, 'noise_variance_'):
        plot_gallery("Pixelwise variance",
                     estimator.noise_variance_.reshape(1, -1), n_col=1,
                     n_row=1)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

plt.show()
'''
    


#######
#Learn labels on the Eigenfaces#######
k = 3000 # number of ratings to choose, low number for fast testing
username = 'sam'
print 'reading training and testing images for user ', username
fdb = facedb.FaceDB(img_dir_path, 'test.db')
ratings = fdb.ratings_for_user(username)[0:k]
paths = []
for r in ratings:
  paths.append(img_dir_path + r[0])

'''
ignoreFirst = 5 #ignore the first x eigenfaces
weight_k = 400 #use this many eigenfaces total
'''

estimator = estimators[0][1]
clf_components_list = [5, 10, 25, 50, 100]
for clf_components in clf_components_list:
  print 'trying with top', clf_components, 'eigenfaces:'
  data = faces
  estimator.fit(data)


  X = estimator.transform(np.array([ open_preprocess(f) for f in paths]))
  y = np.array([r[1] for r in ratings])

  #split X into training set and testing set
  split_ind = 2000
  X_train = X[:split_ind, :clf_components]
  X_test = X[split_ind:,:clf_components]
  y_train = y[:split_ind]
  y_test = y[split_ind:]

  #Positive training examples

  print 'fitting model to training data'
  #Positive examples are more rare, so they should be weighted more
  #clf = svm.SVC(class_weight={0: 1, 1: 100})
  #clf = svm.SVC()
  #clf.fit(X_train, y_train) 
  clf = GaussianNB()
  clf.fit(X_train, y_train) 

  print 'scoring test data'
  predictions = clf.predict(X_test)
  total = [0.0, 0.0] #Total test examples in either category '0' or '1'
  acc = [0.0, 0.0] #accuracy of classifier for categories '0' and '1'
  for i in xrange(len(predictions)):
    total[y_test[i]] += 1
    if predictions[i] == y_test[i]:
      acc[y_test[i]] += 1

  overall_accuracy = sum(acc) / sum(total)
  acc[0] = acc[0] / total[0]
  acc[1] = acc[1] / total[1]
  print 'accuracy for 0:', acc[0]
  print 'accuracy for 1:', acc[1]
  print 'overall accuracy:', overall_accuracy

    

  '''
  X_pos = X_train[y_train==1]

  params = {'bandwidth': np.logspace(-1, 1, 20)}
  grid = GridSearchCV(KernelDensity(), params)
  grid.fit(X_pos)
  print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

  kde = grid.best_estimator_
  samples = kde.sample(10)
  for i in range(samples.shape[0]):
    print samples[i, :].shape
    print estimator.components_.shape
    print np.dot(samples[i, :], estimator.components_).shape
    sample_recon = np.dot(samples[i, :], estimator.components_)
    scipy.misc.imsave('sample_{}.jpg'.format(i), sample_recon.reshape(256,256))

  '''
