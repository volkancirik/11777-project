# -*- coding: utf-8 -*-
import sys, gzip, theano, nltk
import numpy as np
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
"""
prepare data for training / testing
"""
UNK = '*UNKNOWN*'

def open_file(fname):
	try:
		f = open(fname)
	except:
		print >> sys.stderr, "could not open the file %s" % (fname)
		quit(0)
	return f

def process_target(fname, word_idx, tokenize = False, max_len = -1):

	in_file = open_file(fname)

	V = len(word_idx)
	unk = 0.0
	ntok = 0.0

	S = [line.lower().strip().split() for line in in_file]
	N = len(S)

	Y = np.zeros((N,max_len,V), dtype=np.bool)
	for i,s in enumerate(S):
		ntok += len(s)
		for j,tok in enumerate(s):
			try:
				idx = word_idx[tok]
			except:
				unk += 1
				idx = word_idx[UNK]
				pass
			Y[i,j,idx] = 1
	print >> sys.stderr, "UNK rate for %s is %f" % (fname, unk/ntok)
	return Y

def process_source(fname, v_map, dim):

	in_file = open_file(fname)

	unk = 0.0
	ntok = 0.0

	X = []
	for line in in_file:
		s = nltk.word_tokenize(line.lower().strip())
		x = np.zeros((1,len(s), dim), dtype=theano.config.floatX)
		ntok += len(s)
		for j,tok in enumerate(s):
			try:
				v = v_map[tok]
			except:
				unk +=1
				v = v_map[UNK]
				pass
			x[0,j,:] = list(v)
		X += [x]

	print >> sys.stderr, "UNK rate for %s is %f" % (fname, unk/ntok)
	return X

def process_image(fname):
	f = open_file(fname)
	package = pickle.load(f)
	### validate using package['filenames'] and ../data/{train|val}.img
	return package['features']

def prepare_train(path_prefix = '../data/', train_source = 'train.en', train_target = 'train.de', val_source = 'val.en', val_target = 'val.de', wvec = '../embeddings/word2vec.pkl',TRESHOLD = 1, train_img = 'TRAIN.vgg19.fc7.pkl', val_img = 'VAL.vgg19.fc7.pkl'):

	tr_t = open_file(path_prefix + train_target)
	val_t = open_file(path_prefix + val_target)

	try:
		print >> sys.stderr, "loading word vectors..."
		v_map = pickle.load(gzip.open(wvec, "rb"))
		dim = len(list(v_map[UNK]))
		print >> sys.stderr, "%d dim word vectors are loaded.." % (dim)
	except:
		print >> sys.stderr, "word embedding file %s cannot be read." % (wvec)
		quit(1)


	sen_len = []
	for de in [tr_t,val_t]:
		sen_len += [len(line.strip().split()) for line in de]
	max_len = np.max(sen_len)
	tr_t = open_file(path_prefix + train_target)

	vocab_de = defaultdict(int)
	for de in [tr_t]:
		for line in de:
			for tok in line.lower().strip().split():
				vocab_de[tok] +=1

	words = [w for w in vocab_de]
	vocab_de[UNK] = TRESHOLD + 1
	for w in words:
		if vocab_de[w] <= TRESHOLD:
			del vocab_de[w]
	word_idx = dict((c, i) for i, c in enumerate(vocab_de))
	idx_word = dict((i, c) for i, c in enumerate(vocab_de))

	X_tr = process_source(path_prefix + train_source, v_map, dim)
	Y_tr = process_target(path_prefix + train_target, word_idx, max_len = max_len)
	X_tr_img = process_image(path_prefix + train_img)

	X_val = process_source(path_prefix + val_source, v_map, dim)
	Y_val = process_target(path_prefix + val_target, word_idx, max_len = max_len)
	X_val_img = process_image(path_prefix + val_img)
	return X_tr,Y_tr,X_tr_img,X_val,Y_val,X_val_img
