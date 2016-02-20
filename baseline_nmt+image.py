from __future__ import print_function

import numpy as np
import sys
import cPickle as pickle
import json, time, datetime, os

from prepare_data import *
from utils import *
import json, time, datetime, os
import cPickle as pickle

from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge
from keras.layers import recurrent
from keras.optimizers import RMSprop

from prepare_data import prepare_train
from utils import get_parser_nmt
'''
MMMT baseline model - basic enc-dec + FC layer from CNN
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

img_model = Sequential()
img_model.add(Dense(4096, input_shape=(IMG_SIZE,)))
img_model.add(Activation('relu'))
img_model.add(Dropout(0.2))
img_model.add(RepeatVector(MAXLEN))

enc_model = Sequential()
enc_model.add(RNN(HIDDEN_SIZE, input_shape = (None,DIM)))
enc_model.add(RepeatVector(MAXLEN))

model = Sequential()
model.add(Merge([enc_model, img_model], mode = 'concat', concat_axis = -1))

for _ in xrange(LAYERS):
	model.add(RNN(HIDDEN_SIZE, return_sequences=True))
	model.add(TimeDistributedDense(HIDDEN_SIZE))
	model.add(Activation('relu'))

model.add(TimeDistributedDense(V))
model.add(Activation('softmax'))
print("compiling model...")
optimizer = RMSprop(clipnorm = 5)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', class_mode='categorical')

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
		eh = model.fit([x,x_img], y, batch_size = 1, nb_epoch=1, show_accuracy = False, verbose = False)
		e_loss += eh.history['loss']

	e_val_loss = []
	## validate
	for i in xrange(len(X_val)):
		x = X_val[i]
		x_img = X_tr_img[i].reshape((1,IMG_SIZE))
		y = Y_val[i].reshape((1,MAXLEN,V))
		v_loss = model.evaluate([x,x_img], y, batch_size = 1, show_accuracy = False, verbose = False)
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
