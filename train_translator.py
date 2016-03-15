from __future__ import print_function

import json, time, datetime, os, sys
import cPickle as pickle
import numpy as np

from get_model import get_model
from prepare_data import prepare_train
from utils import get_parser_nmt
'''
train a MMMT model
'''
### get arguments
parser = get_parser_nmt()
p = parser.parse_args()

### Parameters
MINI_BATCH = True
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
MODEL = p.model
PATIENCE = p.patience
HIDDEN_SIZE = p.n_hidden
LAYERS = p.layers
BATCH_SIZE = p.batch_size
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'M' + str(MODEL) + '_U' + p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS)

### get data
X_tr, Y_tr, X_tr_img, X_val, Y_val, X_val_img, word_idx, idx_word = prepare_train(mini_batch = MINI_BATCH)

IMG_SIZE = X_tr_img.shape[1]
V = Y_tr.shape[2]
N = len(X_tr)
MAXLEN = Y_tr.shape[1]
if MINI_BATCH:
	DIM = X_tr.shape[2]
else:
	DIM = X_tr[0].shape[2]

model = get_model(MODEL, p.unit, IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE)
pat = 0
train_history = {'loss' : [], 'val_loss' : []}
best_val_loss = float('inf')

print("saving stuff...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)
pickle.dump({'word_idx' : word_idx,'idx_word' : idx_word, 'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))

print("training model...")
for iteration in xrange(EPOCH):
	print('_' * 50)
	print('iteration {}/{}'.format(iteration+1,EPOCH))

	if MINI_BATCH:
		eh = model.fit({'input_en' : X_tr, 'input_img' : X_tr_img, 'output' : Y_tr}, batch_size = BATCH_SIZE, nb_epoch=1, verbose = True, validation_data = {'input_en' : X_val, 'input_img' : X_val_img, 'output' : Y_val})

		for key in ['loss','val_loss']:
			train_history[key] += eh.history[key]
	else:
		e_loss = []
		for i in xrange(N):
			x = X_tr[i]
			x_img = X_tr_img[i].reshape((1,IMG_SIZE))
			y = Y_tr[i].reshape((1,MAXLEN,V))
			eh = model.fit({'input_en' : x, 'input_img' : x_img, 'output' : y}, batch_size = 1, nb_epoch=1, verbose = False)
			e_loss += eh.history['loss']

		e_val_loss = []
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

pickle.dump({'word_idx' : word_idx,'idx_word' : idx_word, 'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))
