# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import defaultdict

def find_bucket(bucket_length, value):
	idx = (np.abs(np.array(bucket_length)-value)).argmin()
	if bucket_length[idx] < value:
		idx +=1
	return idx

def distribute_buckets(length, X_list, Y_list, step_size, x_set, y_set, verbose = False):
	bucket_length = range(np.min(length),np.max(length),step_size) + [np.max(length)]
	bucket_length = bucket_length[1:]

	bucket_size = defaultdict(int)
	diffs = defaultdict(list)
	for l in length:
		bucket_size[find_bucket(bucket_length,l)] += 1
		diffs[find_bucket(bucket_length,l)] += [bucket_length[find_bucket(bucket_length,l)] - l]
	if verbose:
		for key in diffs:
			print >> sys.stderr, bucket_length[key],"-->",diffs[key]

	bucket_x = []
	bucket_y = []
	bucket_count = defaultdict(int)

	for b_idx,bucket in enumerate(bucket_length):
		new_x = []
		for i,x_orig in enumerate(X_list):
			shape = list(x_orig.shape)
			shape[0] = bucket_size[b_idx]
			if i in x_set:
				shape[1] = bucket
			x_new = np.zeros(shape, dtype = x_orig.dtype)
			new_x += [x_new]
		bucket_x += [new_x]
		new_y = []
		for i,y_orig in enumerate(Y_list):
			shape = list(y_orig.shape)
			shape[0] = bucket_size[b_idx]
			if i in y_set:
				shape[1] = bucket
			y_new = np.zeros(shape, dtype = x_orig.dtype)
			new_y += [y_new]
		bucket_y += [new_y]

	max_len = np.max(length)
	for i in xrange(len(length)):
		idx = find_bucket(bucket_length, length[i])
		for j,x_orig in enumerate(X_list):
			if j in x_set:
				bucket_x[idx][j][ bucket_count[idx], bucket_length[idx] - length[i]: ] = x_orig[i, max_len - length[i]:] ## remember padding left
			else:
				bucket_x[idx][j][ bucket_count[idx]] = x_orig[i]

		for j,y_orig in enumerate(Y_list):
			if j in y_set:
				bucket_y[idx][j][ bucket_count[idx], bucket_length[idx] - length[i]:] = y_orig[i, max_len - length[i]:]
			else:
				bucket_y[idx][j][ bucket_count[idx]] = y_orig[i]
		bucket_count[idx] += 1

	if verbose:
		for i in xrange(len(bucket_length)):
			print >> sys.stderr, bucket_size[i], bucket_count[i]

	print >> sys.stderr, " ".join([ str(key)+'_'+str(bucket_count[key])+'_'+str(bucket_length[key]) for key in bucket_count ])
	print >> sys.stderr, "%d buckets are created from %d to %d with step-size %d" % (len(bucket_length),bucket_length[0],bucket_length[-1],step_size)

	return bucket_x, bucket_y

def unit_test(min_l, max_l, n, dim = 2, step_size = 3):
	np.random.seed(10)
	length = [np.random.choice(range(min_l,max_l)) for i in xrange(n)]
	max_len = np.max(length)
	X_1 = np.zeros((n,max_len, dim,dim), dtype = float)
	X_2 = np.zeros((n,dim), dtype = float)

	Y_2 = np.zeros((n,max_len,dim,dim,dim), dtype = bool)
	Y_1 = np.zeros((n,dim*2), dtype = bool)

	for i,l in enumerate(length):
		for j in xrange(l):
			x_1 = np.random.random((dim,dim))
			y_2 = np.random.choice([True, False],size = (dim,dim,dim))
			X_1[i,j,:] = x_1
			Y_2[i,j,:] = y_2

		x_2 = np.random.random((dim))
		y_1 = np.random.choice([True, False],size = (dim*2))

		X_2[i] = x_2
		Y_1[i] = y_1
	return X_1,X_2,Y_1,Y_2,length

if __name__ == '__main__':
	X_1,X_2,Y_1,Y_2,length =unit_test(2,10,6)
	print length
	print X_1
	print "_" * 10
	print Y_2
	print "_" * 50

	bucket_x, bucket_y = distribute_buckets(length, [X_1,X_2], [Y_1,Y_2], 4, set([0]),set([1]))
	for tuple_x in bucket_x:
		for x in tuple_x:
			print x.shape
		print "_"*10

	print "X"*50
	for tuple_y in bucket_y:
		for y in tuple_y:
			print y.shape
		print "_"*10
