from __future__ import print_function

import json, time, datetime, os, sys
import cPickle as pickle
import numpy as np

from get_model import get_model
from prepare_data import prepare_train
from utils import get_parser_nmt, decode_predicted
from buckets import distribute_buckets
'''
train a MMMT model
'''
### get arguments

parser = get_parser_nmt()
p = parser.parse_args()

### Parameters
MINI_BATCH = True
HIERARCHICAL = p.hierarchical
BATCH_SIZE = p.batch_size
EPOCH = p.n_epochs
MODEL = p.model
PATIENCE = p.patience
HIDDEN_SIZE = p.n_hidden
LAYERS = p.layers
DROPOUT = p.dropout
BATCH_SIZE = p.batch_size
SOURCE = p.source + p.suffix
PREFIX = 'exp/'+p.prefix + '/'
SUFFIX = p.suffix
REPEAT= {'full' : True, 'truncated' : False, 'debug' : False, 'task1' : False}[SUFFIX]
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'M' + str(MODEL) + '_U' + p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_HIER' + str(HIERARCHICAL) + '_SUF' + SUFFIX + '_DR' + str(DROPOUT)

### get data
X_tr, [Y_tr, Y_tr_shifted] , X_tr_img, X_val, [Y_val,Y_val_shifted], X_val_img, dicts , [length_tr, length_val] = prepare_train(use_hierarchical = HIERARCHICAL, suffix = {'full' : '.all.tokenized.unkified', 'truncated' : '.truncated', 'debug' : '.debug', 'task1' : '.task1'}[SUFFIX], repeat = REPEAT, model_type = MODEL)

V_en = len(dicts['word_idx_en'])
V_de = len(dicts['word_idx_de'])

IMG_SIZE = X_tr_img.shape[1]
N = len(X_tr)
if HIERARCHICAL:
	Y_tr_1, Y_tr_2, Y_tr_3, Y_tr_4 = Y_tr
	Y_val_1, Y_val_2, Y_val_3, Y_val_4 = Y_val
	MAXLEN = Y_tr_1.shape[1]
else:
	MAXLEN = Y_tr.shape[1]

b_X_tr, b_Y_tr = distribute_buckets(length_tr, [X_tr,X_tr_img], Y_tr, step_size = 10, x_set = set([0]), y_set = set())

model = get_model(MODEL, p.unit, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, use_hierarchical = HIERARCHICAL)
pat = 0

train_history = {'loss' : [], 'val_meteor' : []}
best_val_meteor = 0

print("saving stuff...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)
pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))

print("training model with {} parameters...".format(model.get_n_params()))
NB = len(b_X_tr)
for iteration in xrange(EPOCH):
	print('_' * 50)

	train_history['loss'] += [0]
	for j in xrange(NB):
		[X_tr, X_tr_img] = b_X_tr[j]
		[Y_tr_1, Y_tr_2, Y_tr_3, Y_tr_4] = b_Y_tr[j]
		print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, j+1,NB))

		eh = model.fit({'input_en' : X_tr , 'input_img' : X_tr_img,  'output_1' : Y_tr_1, 'output_2' : Y_tr_2, 'output_3' : Y_tr_3, 'output_4' : Y_tr_4}, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = True)

		for key in ['loss']:
			train_history[key][-1] += eh.history[key][0]

	predicted = model.predict({'input_en' : X_val, 'input_img' : X_val_img}, batch_size = BATCH_SIZE)
	decode_predicted(PREFIX + FOOTPRINT + '.predicted', predicted, dicts)
	cmd = 'java -Xmx2G -jar ../bin/meteor-1.5/meteor-1.5.jar %s %s -l de > %s' % (PREFIX + FOOTPRINT + '.predicted', SOURCE,PREFIX + FOOTPRINT + '.meteor')
	os.system(cmd)
	cmd = 'grep Final %s | cut -d ":" -f2 | awk "{print $1}" > %s' % (PREFIX + FOOTPRINT + '.meteor',PREFIX + FOOTPRINT + '.meteor.accuracy')
	os.system(cmd)
	epoch_val_meteor = float(open(PREFIX + FOOTPRINT + '.meteor.accuracy').readlines()[0].strip())
	train_history['val_meteor'] += [epoch_val_meteor]

	print("TR  {} VAL {} best VL {} no improvement in {}".format(train_history['loss'][-1],train_history['val_meteor'][-1],best_val_meteor,pat))

	if train_history['val_meteor'][-1] <= best_val_meteor:
		pat += 1
	else:
		pat = 0
		best_val_meteor = train_history['val_meteor'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	if pat == PATIENCE:
		break

pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))
