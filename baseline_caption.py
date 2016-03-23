# -*- coding: utf-8 -*-
from __future__ import print_function

import json, time, datetime, os, sys
import numpy as np
import cPickle as pickle

from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, MaskedLayer
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop

from prepare_data import prepare_train
from utils import get_parser_nmt
'''
MMMT baseline model - basic enc-dec for MT
'''
UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU , 'lstm' : recurrent.LSTM}

### get arguments
parser = get_parser_nmt()
p = parser.parse_args()

### Parameters
RNN = UNIT[p.unit]
HIERARCHICAL = p.hierarchical
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
MODEL = p.model
PATIENCE = p.patience
DROPOUT = p.dropout 
HIDDEN_SIZE = p.n_hidden
LAYERS = p.layers
BATCH_SIZE = p.batch_size
PREFIX = 'exp/'+p.prefix + '/'
SUFFIX = p.suffix
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'U' + p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_HIER' + str(HIERARCHICAL) + '_D' + str(DROPOUT) + '_SUF' + SUFFIX

### get data
_, Y_tr, X_tr_img , _ , Y_val, X_val_img , dicts, _ = prepare_train(use_hierarchical = HIERARCHICAL, repeat = False, suffix = {'full' : '.all.tokenized.unkified', 'truncated' : '.truncated', 'debug' : '.debug'}[SUFFIX])

V_de = len(dicts['word_idx_de'])

IMG_SIZE = X_tr_img.shape[1]
if HIERARCHICAL:
	Y_tr_1, Y_tr_2, Y_tr_3, Y_tr_4 = Y_tr
	Y_val_1, Y_val_2, Y_val_3, Y_val_4 = Y_val
	MAXLEN = Y_tr_1.shape[1]
	N = Y_tr_1.shape[0]
else:
	MAXLEN = Y_tr.shape[1]
	N = Y_tr.shape[0]

print('building model...')
model = Graph()
model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))

prev_layer = 'input_img'
for layer in xrange(LAYERS):
	model.add_node(Dense(HIDDEN_SIZE, activation = 'relu'), name = 'dense'+str(layer), input = prev_layer)
	model.add_node(Dropout(DROPOUT),name = 'dense'+str(layer)+'_d', input = 'dense'+str(layer))
	prev_layer = 'dense'+str(layer)+'_d'

model.add_node(RepeatVector(MAXLEN), name = 'rv', input = prev_layer)
prev_layer = 'rv'
for layer in xrange(LAYERS):
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'decoder_rnn'+str(layer), input = prev_layer)
	model.add_node(Dropout(DROPOUT),name = 'decoder_rnn'+str(layer)+'_d', input = 'decoder_rnn'+str(layer))
	prev_layer = 'decoder_rnn'+str(layer)+'_d'

optimizer = RMSprop(clipnorm = 5)
if HIERARCHICAL:
	DIM = int(pow(V_de,0.25)) + 1
	model.add_node(TimeDistributedDense(DIM), name = 'tdd_1', input = prev_layer)
	model.add_node(TimeDistributedDense(DIM), name = 'tdd_2', input = prev_layer)
	model.add_node(TimeDistributedDense(DIM), name = 'tdd_3', input = prev_layer)
	model.add_node(TimeDistributedDense(DIM), name = 'tdd_4', input = prev_layer)

	model.add_node(Activation('softmax'), name = 'softmax_1', input = 'tdd_1')
	model.add_node(Activation('softmax'), name = 'softmax_2', input = 'tdd_2')
	model.add_node(Activation('softmax'), name = 'softmax_3', input = 'tdd_3')
	model.add_node(Activation('softmax'), name = 'softmax_4', input = 'tdd_4')

	model.add_output(name = 'output_1', input = 'softmax_1')
	model.add_output(name = 'output_2', input = 'softmax_2')
	model.add_output(name = 'output_3', input = 'softmax_3')
	model.add_output(name = 'output_4', input = 'softmax_4')

	model.compile(loss = { 'output_1' : 'categorical_crossentropy', 'output_2' : 'categorical_crossentropy', 'output_3' : 'categorical_crossentropy', 'output_4' : 'categorical_crossentropy'}, optimizer= optimizer)
else:
	model.add_node(TimeDistributedDense(V_de), name = 'tdd', input = prev_layer)
	model.add_node(Dropout(DROPOUT),name = 'tdd_d', input = 'tdd')
	model.add_node(Activation('softmax'), name = 'softmax', input = 'tdd_d')
	model.add_output(name = 'output', input = 'softmax')
	model.compile(loss = { 'output' : 'categorical_crossentropy'}, optimizer= optimizer)

pat = 0
train_history = {'loss' : [], 'val_loss' : []}
best_val_loss = float('inf')

print("saving stuff...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)
pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))

print("training model with {} parameters...".format(model.get_n_params()))
for iteration in xrange(EPOCH):
	print('_' * 50)
	print('iteration {}/{}'.format(iteration+1,EPOCH))

	if HIERARCHICAL:
		eh = model.fit({'input_img' : X_tr_img, 'output_1' : Y_tr_1, 'output_2' : Y_tr_2, 'output_3' : Y_tr_3, 'output_4' : Y_tr_4}, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = True, validation_data = {'input_img' : X_val_img, 'output_1' : Y_val_1, 'output_2' : Y_val_2, 'output_3' : Y_val_3, 'output_4' : Y_val_4})
	else:
		eh = model.fit({'input_img' : X_tr_img, 'output' : Y_tr}, batch_size = BATCH_SIZE, nb_epoch=1, verbose = True, validation_data = {'input_img' : X_val, 'output' : Y_val})

	for key in ['loss','val_loss']:
		train_history[key] += eh.history[key]

	print("TR  {} VAL {} best VL {} no improvement in {}".format(train_history['loss'][-1],train_history['val_loss'][-1],best_val_loss,pat))

	if train_history['val_loss'][-1] > best_val_loss:
		pat += 1
	else:
		pat = 0
		best_val_loss = train_history['val_loss'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	if pat == PATIENCE:
		break

pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))
