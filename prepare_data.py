# -*- coding: utf-8 -*-
import sys, gzip, nltk
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

"""
prepare data for training / testing
"""
UNK = '*UNKNOWN*'
EOS = '</s>'
def open_file(fname):
	try:
		f = open(fname)
	except:
		print >> sys.stderr, "could not open the file %s" % (fname)
		quit(0)
	return f

def process_target(fname, word_idx, tokenize = False, max_len = -1, use_hierarchical = True): ### shifted should be here!

	in_file = open_file(fname)
	V = len(word_idx)
	dim = int(pow(V,0.25)) + 1
	V = dim ** 4

	S = [line.strip().split() + [EOS] for line in in_file]
	N = len(S)

	if use_hierarchical:
		print >> sys.stderr, "using hierarchical softmax with %d dimensional vectors " % (dim)
		Y_1 = np.zeros((N,max_len, dim), dtype=np.bool)
		Y_2 = np.zeros((N,max_len, dim), dtype=np.bool)
		Y_3 = np.zeros((N,max_len, dim), dtype=np.bool)
		Y_4 = np.zeros((N,max_len, dim), dtype=np.bool)
	else:
		Y = np.zeros((N,max_len,V), dtype=np.bool)

#	Yshifted = np.zeros((N,max_len,V), dtype=np.bool)
	Yshifted = np.zeros((N,max_len), dtype = 'int64')
	for i,s in enumerate(S):
		for j,tok in enumerate(s):
			idx = word_idx[tok]

			if use_hierarchical:
				idx_1 = idx / (dim**3)
				idx_2 = (idx - (dim**3)*idx_1) / (dim**2)
				idx_3 = (idx - (dim**3)*idx_1 - (dim**2)*idx_2) / (dim)
				idx_4 = idx % (dim)

				Y_1[i, j, idx_1 ] = 1
				Y_2[i, j, idx_2 ] = 1
				Y_3[i, j, idx_3 ] = 1
				Y_4[i, j, idx_4 ] = 1

			else:
				Y[i, j, idx ] = 1
			if j+1 < len(s): ###
#				Yshifted[i, j+1, idx ] = 1
				Yshifted[i, j+1] = idx

	if use_hierarchical:
		Y = Y_1,Y_2,Y_3,Y_4
	return [Y, Yshifted]

def process_source(fname, word_idx_en, max_len = -1):

	in_file = open_file(fname)
	S = [line for line in in_file]
	N = len(S)
	X = np.zeros((N,max_len), dtype = 'int64')

	length = []
	for i,line in enumerate(S):
		s = line.strip().split() + [EOS]
		pad = max_len - len(s) #
		for j,tok in enumerate(s):
			X[i,pad + j] = word_idx_en[tok] #
		length += [len(s)]
	return X, length

def process_image(fname, cnn_filter_file, repeat = False, mode = 0):
	if mode <= 1:
		f = open_file(fname)
		package = pickle.load(f)
		if len(package['features'].shape) == 4:
			feats_cnn =  package['features'].reshape(package['features'].shape[0],package['features'].shape[1],-1)
		else:
			feats_cnn =  package['features']
	if mode >= 1:
		f = open_file(cnn_filter_file)
		package = pickle.load(f)
		feats_filter = np.concatenate((package['o1'],package['o2'],package['o3'],package['o4']), axis = 1)
	if mode == 1:
		feats = np.concatenate((feats_cnn,feats_filter), axis = 1)
	elif mode == 0:
		feats = feats_cnn
	elif mode == 2:
		feats = feats_filter
	else:
		raise NotImplementedError()
	return feats

def get_max_len(f_list):
	sen_len = []
	for f in f_list:
		sen_len += [ len(line.strip().split()) + 1 for line in f]
	return np.max(sen_len)

def get_dicts(f_list):

	vocab = defaultdict(int)
	for f_name in f_list:
		f = open_file(f_name)
		for line in f:
			for tok in line.strip().split():
				vocab[tok] +=1

	vocab[UNK] = 1
	vocab[EOS] = 1
	word_idx = dict((c, i) for i, c in enumerate(vocab))
	idx_word = dict((i, c) for i, c in enumerate(vocab))

	V = len(word_idx)
	first_w = idx_word[0]
	idx_word[0] = '*dummy*'
	idx_word[V] = first_w
	word_idx[first_w] = V
	word_idx['*dummy*'] = 0

	return word_idx, idx_word

def prepare_train(path_prefix = '../data/', train_source = 'train.en', train_target = 'train.de', val_source = 'val.en', val_target = 'val.de', TRESHOLD = 1, train_img = 'TRAIN.vgg19.fc7.pkl', val_img = 'VAL.vgg19.fc7.pkl', use_hierarchical = True, repeat = False, suffix = '.all.tokenized.unkified', model_type = 0, mode = 0, cnn_filter_val = '../cnn_filter/exp/BASELINE0/H256_L2_DR0.0_Arelu.feats.val.pkl', cnn_filter_train = '../cnn_filter/exp/BASELINE0/H256_L2_DR0.0_Arelu.feats.train.pkl'):
	if model_type == 2:
		train_img = 'TRAIN.vgg19.conv5_4.pkl'
		val_img = 'VAL.vgg19.conv5_4.pkl'

	if suffix == '.debug':
		train_target = val_target
		train_source = val_source
		train_img = val_img

	tr_t = open_file(path_prefix + train_target + suffix)
	val_t = open_file(path_prefix + val_target + suffix)

	tr_s = open_file(path_prefix + train_source + suffix)
	val_s = open_file(path_prefix + val_source + suffix)

	max_len_en = get_max_len([tr_s,val_s])
	max_len_de = get_max_len([tr_t,val_t])

	word_idx_en, idx_word_en = get_dicts([path_prefix + train_source + suffix, path_prefix + val_source + suffix])
	word_idx_de, idx_word_de = get_dicts([path_prefix + train_target + suffix, path_prefix + val_target + suffix])

	print >> sys.stderr, "vocabulary size for english %d and for german is %d \n max seq len for english %d and for german is %d" % (len(idx_word_en),len(idx_word_de),max_len_en, max_len_de)

	X_tr, length_tr = process_source(path_prefix + train_source  + suffix, word_idx_en, max_len = max_len_en)
	[Y_tr, Y_tr_shifted] = process_target(path_prefix + train_target + suffix, word_idx_de, max_len = max_len_de, use_hierarchical = use_hierarchical)
	if model_type != 3:
		X_tr_img = process_image(path_prefix + train_img, cnn_filter_train, repeat = repeat, mode = mode)
	else:
		X_tr_img = []

	X_val, length_val = process_source(path_prefix + val_source + suffix, word_idx_en,  max_len = max_len_en)
	[Y_val,Y_val_shifted] = process_target(path_prefix + val_target + suffix, word_idx_de, max_len = max_len_de, use_hierarchical = use_hierarchical)
	if model_type != 3:
		X_val_img = process_image(path_prefix + val_img, cnn_filter_val, repeat = repeat, mode = mode)
	else:
		X_val_img = []

	dicts = {'word_idx_en' :  word_idx_en, 'idx_word_en' : idx_word_en, 'word_idx_de' :  word_idx_de, 'idx_word_de' : idx_word_de}

	if suffix == '.debug':
		X_tr_img = X_tr_img[:X_tr.shape[0]]
		X_val_img = X_tr_img

	return X_tr, [Y_tr, Y_tr_shifted] , X_tr_img, X_val, [Y_val,Y_val_shifted], X_val_img, dicts , [length_tr, length_val]

def prepare_test(word_idx_en, word_idx_de, path_prefix = '../data/', test_source = 'val.en', test_target = 'val.de', test_img = 'VAL.vgg19.fc7.pkl', repeat = False, suffix = '.all.tokenized.unkified'):

	test_s = open_file(path_prefix + test_source + suffix)
	test_t = open_file(path_prefix + test_target + suffix)

	max_len_en = get_max_len([test_s])
	max_len_de = get_max_len([test_t])

	X_test, length_test = process_source(path_prefix + test_source + suffix, word_idx_en, max_len = max_len_en)
	X_test_img = process_image(path_prefix + test_img, repeat = repeat)
	if suffix == '.debug':
		X_test_img = X_test_img[:X_test.shape[0]]

	[Y_test, Y_test_shifted] = process_target(path_prefix + test_target + suffix, word_idx_de, max_len = max_len_de, use_hierarchical = True)

	return X_test, test_t, X_test_img ,[Y_test, Y_test_shifted]
if __name__ == '__main__':
	cnn_filter_val = '../cnn_filter/exp/BASELINE0/H256_L2_DR0.0_Arelu.feats.val.pkl'
	val_img = '../data/VAL.vgg19.fc7.pkl'
	for mode in range(3):
		feats = process_image(val_img, cnn_filter_val, mode = mode)
		print feats.shape

