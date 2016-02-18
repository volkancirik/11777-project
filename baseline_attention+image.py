from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import json, time, datetime, os

from prepare_data import *
from utils import *
import json, time, datetime, os
import cPickle as pickle

from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge
from keras.layers import recurrent
from keras.optimizers import RMSprop

from keras.layers.attention import DenseAttention, TimeDistributedAttention

from prepare_data import prepare_train
from utils import get_parser_nmt
'''
MMMT baseline model - attention on source language + FC layer from CNN
'''
UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}

### get arguments
parser = get_parser_nmt()
p = parser.parse_args()

### Parameters
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
LAYERS = p.layers
RNN = UNIT[p.unit]
PATIENCE = p.patience
HIDDEN_SIZE = p.n_hidden
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'U'+p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS)

### get data
X_tr,Y_tr,X_tr_img,X_val,Y_val,X_val_img = prepare_train()

IMG_SIZE = X_tr_img.shape[1]
V = Y_tr.shape[2]
N = len(X_tr)
MAXLEN = Y_tr.shape[1]
DIM = X_tr[0].shape[2]

print('building model...')
model = Graph()
model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
model.add_input(name = 'input_en', input_shape = (None,DIM))

model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='encoder_context', input='input_en')
model.add_node(RepeatVector(MAXLEN), name='recurrent_context', input='input_img')

model.add_node(TimeDistributedAttention(prev_dim = IMG_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True), name='attention', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')
model.add_node(TimeDistributedDense(V), name='tdd', input= 'attention')
model.add_node(Activation('softmax'), name = 'softmax',input = 'tdd')
model.add_output(name='output', input='softmax')
optimizer = RMSprop(clipnorm = 5)
print("compiling model...")
model.compile(optimizer = optimizer, loss = {'output': 'categorical_crossentropy'})

### save architecture
print("saving architecture...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
    json.dump(model.to_json(), outfile)

print("training model...")

pat = 0
train_history = {'loss' : [], 'val_loss' : []}
best_val_loss = float('inf')

for iteration in xrange(EPOCH):

	print('_' * 50)
	print('iteration {}/{}'.format(iteration+1,EPOCH))

	e_loss = []
	for i in xrange(N):
		x = X_tr[i]
		x_img = X_tr_img[i].reshape((1,IMG_SIZE))
		y = Y_tr[i].reshape((1,MAXLEN,V))
		eh = model.fit({'input_en' : x, 'input_img' : x_img, 'output' : y}, batch_size = 1, nb_epoch=1, verbose = False)
		e_loss += eh.history['loss']

	e_val_loss = []
	## validate
	for i in xrange(len(X_val)):
		x = X_val[i]
		x_img = X_tr_img[i].reshape((1,IMG_SIZE))
		y = Y_val[i].reshape((1,MAXLEN,V))
		v_loss = model.evaluate({'input_en' : x, 'input_img' : x_img, 'output' : y}, batch_size = 1, verbose = False)
		e_val_loss += [v_loss]

	train_history['loss'] += [np.mean(e_loss)]
	train_history['val_loss'] += [np.mean(e_val_loss)]

	print("TR  {} VAL {} best VL {} no improvement in {}".format(train_history['loss'][-1],train_history['val_loss'][-1],best_val_loss,pat))
	if train_history['val_loss'][-1] > best_val_loss:
		pat += 1
	else:
		pat = 0
		best_val_loss = train_history['val_loss'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	if pat == PATIENCE:
		break
