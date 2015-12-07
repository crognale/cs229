import os
from PIL import Image #pip install pillow on osx

import numpy as np
import scipy
import scipy.misc

from os.path import isfile

n = 1000 #number of faces to use for generating eigenfaces
h = 256
max_w = 256


basedir = '10kfaces/'

def open_preprocess(f):
	img = scipy.misc.imread(f, True)
	if img.shape[0] is not h:
		img = scipy.misc.imresize(img, float(h) / img.shape[0]);
	w = img.shape[1]

	padding = max_w - w
	pad_left = padding / 2
	pad_right = (padding / 2) + padding % 2
	#print f, img.shape, pad_left, pad_right

	left_padding = np.ones((h, pad_left)) * 255.0
	right_padding = np.ones((h, pad_right)) * 255.0
	img_arr = np.concatenate((np.concatenate((left_padding, img), axis=1), right_padding), axis=1).flatten()
	return img_arr

if isfile('U.npy') and isfile('S.npy') and isfile('V.npy'):
	print 'Loading U, S, and V from file'
	U = np.load('U.npy')
	S = np.load('S.npy')
	V = np.load('V.npy')
	mu = np.load('mu.npy')
else:
	files = os.listdir(basedir)

	arr = np.zeros((n, h*max_w))
	cur_n = 0
	print 'reading images'
	for f in files[0:n]:
		target_file = basedir + str(f)
		arr[cur_n] = open_preprocess(target_file)
		cur_n += 1

	mu = arr.mean(0)
	X = arr - mu

	print 'doing svd'
	U, S, V = np.linalg.svd(X.transpose(), full_matrices=False)

	print 'saving eigenfaces'
	np.save('U.npy',U)
	np.save('S.npy',S)
	np.save('V.npy',V)
	np.save('mu.npy', mu)

	weights = np.dot(X, U)
	for i in xrange(n):
		scipy.misc.imsave('efaces/{}.png'.format(i), U[:,i].reshape(max_w, h))
		recon = mu + np.dot(weights[i, :], U.T)
		scipy.misc.imsave('efaces/recon{}.png'.format(i), recon.reshape(max_w, h))


print 'reconstructing test image'
ks = [10, 20, 50, 100, 200, 400, 600]
test_X = open_preprocess('testimg.jpg') - mu
test_w = np.dot(test_X, U)
print test_w
#test_recon = mu + np.dot(test_w, U.T)
for k in ks:
	test_recon = mu + np.dot(test_w[0:k], U[:,0:k].T)
	scipy.misc.imsave('testrecon_{}.jpg'.format(k), test_recon.reshape(max_w, h))

print 'done'

