import os
from PIL import Image #pip install pillow on osx

import numpy as np
import scipy
import scipy.misc

from os.path import isfile




def open_preprocess(f):
	img = scipy.misc.imread(f, True)
	if img.shape[0] is not h:
		img = scipy.misc.imresize(img, float(h) / img.shape[0]);
	if img.shape[1] > max_w:
		img = scipy.misc.imresize(img, (h, max_w))
	w = img.shape[1]

	padding = max_w - w
	pad_left = padding / 2
	pad_right = (padding / 2) + padding % 2
	#print f, img.shape, pad_left, pad_right

	left_padding = np.ones((h, pad_left)) * 255.0
	right_padding = np.ones((h, pad_right)) * 255.0
	img_arr = np.concatenate((np.concatenate((left_padding, img), axis=1), right_padding), axis=1).flatten()
	return img_arr

h = 256
max_w = 256

# Class for generating eigenfaces and obtaining singular values for test images.
# Generates eigenfaces using the first n images of the given database. Test images
# can then be decomposed with weights_for_img(). 
class Eigen:
	def __init__(self, img_dir_path,n=1000, forceRefresh=False):
		if isfile('U.npy') and isfile('S.npy') and isfile('V.npy') and not forceRefresh:
			print 'Loading U, S, and V from file'
			self.U = np.load('U.npy')
			self.S = np.load('S.npy')
			self.V = np.load('V.npy')
			self.mu = np.load('mu.npy')
		else:
			files = os.listdir(img_dir_path)

			arr = np.zeros((n, h*max_w))
			cur_n = 0
			print 'reading images'
			for f in files[0:n]:
				target_file = basedir + str(f)
				arr[cur_n] = open_preprocess(target_file)
				cur_n += 1

			self.mu = arr.mean(0)
			X = arr - self.mu

			print 'doing svd'
			self.U, self.S, self.V = np.linalg.svd(X.transpose(), full_matrices=False)

			print 'saving eigenfaces'
			np.save('U.npy',self.U)
			np.save('S.npy',self.S)
			np.save('V.npy',self.V)
			np.save('mu.npy', self.mu)

			'''
			weights = np.dot(X, self.U)
			for i in xrange(n):
				scipy.misc.imsave('efaces/{}.png'.format(i), self.U[:,i].reshape(max_w, h))
				recon = self.mu + np.dot(weights[i, :], self.U.T)
				scipy.misc.imsave('efaces/recon{}.png'.format(i), recon.reshape(max_w, h))
			'''


		'''
		print 'reconstructing test image'
		ks = [10, 20, 50, 100, 200, 400, 600]
		test_X = open_preprocess('testimg.jpg') - self.mu
		test_w = np.dot(test_X, self.U)
		#test_recon = mu + np.dot(test_w, U.T)
		for k in ks:
			test_recon = self.mu + np.dot(test_w[0:k], self.U[:,0:k].T)
			scipy.misc.imsave('testrecon_{}.jpg'.format(k), test_recon.reshape(max_w, h))
		'''
		print 'done'

	def weights_for_img(self, path, ignoreFirst, k):
		X = open_preprocess(path) - self.mu
		return np.dot(X, self.U[:,ignoreFirst:ignoreFirst+k]).T

	def weights_for_imgs(self, paths, ignoreFirst, k):
		X = [open_preprocess(path) - self.mu for path in paths]
		return np.dot(X, self.U[:,ignoreFirst:ignoreFirst+k]).T


