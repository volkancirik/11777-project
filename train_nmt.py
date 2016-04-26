from __future__ import print_function

from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, MaskedLayer, Merge, Layer
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop

from keras.layers.attention import DenseAttention, TimeDistributedAttention
from keras.layers.decoder import LSTMhdecoder, SplitDecoder, LSTMAttentionDecoder, SequenceLayerMerge
from keras.regularizers import l1l2, l2

from keras.layers import dropoutrnn
from keras.layers.dropmodality import DropModality

UNIT = { 'gru' : dropoutrnn.DropoutGRU, 'lstm' : dropoutrnn.DropoutLSTM}
CLIP = 10

import json, time, datetime, os, sys
import cPickle as pickle
import numpy as np

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
DROPMODALITY = p.dropmodality
BATCH_SIZE = p.batch_size
SOURCE = p.source + p.suffix
FILTER_MODE = p.filter_mode
PREFIX = 'exp/'+p.prefix + '/'
SUFFIX = p.suffix
REPEAT= {'full' : True, 'truncated' : False, 'debug' : False, 'task1' : False}[SUFFIX]
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'M' + str(MODEL) + '_U' + p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '_HIER' + str(HIERARCHICAL) + '_SUF' + SUFFIX + '_DR' + str(DROPOUT) + '_FMODE' + str(FILTER_MODE) + '_DM' + str(DROPMODALITY)
RNN = UNIT[p.unit]

### get data
X_tr, [Y_tr, Y_tr_shifted] , X_tr_img, X_val, [Y_val,Y_val_shifted], X_val_img, dicts , [length_tr, length_val] = prepare_train(use_hierarchical = HIERARCHICAL, suffix = {'full' : '.all.tokenized.unkified', 'truncated' : '.truncated', 'debug' : '.debug', 'task1' : '.task1'}[SUFFIX], repeat = REPEAT, model_type = 3)

MAXLEN = Y_tr_shifted.shape[1]
V_en = len(dicts['word_idx_en'])

N = len(X_tr)
Y_tr_1, Y_tr_2, Y_tr_3, Y_tr_4 = Y_tr
Y_val_1, Y_val_2, Y_val_3, Y_val_4 = Y_val
V_de = Y_tr_1.shape[-1] ** 4
DIM = int(pow(V_de,0.25))

print("maxlen & v_en & v_de",MAXLEN,V_en,V_de)

b_X_tr, b_Y_tr = distribute_buckets(length_tr, [X_tr], list(Y_tr) +[Y_tr_shifted], step_size = 5, x_set = set([0]), y_set = set())

model = Graph()

model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
model.add_input(name = 'input_de', input_shape = (None,), dtype = 'int64')

model.add_node(Layer(),name = 'shifted_de', input = 'input_de')
model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

prev_layer = 'embedding'
for layer in xrange(LAYERS -1):
	model.add_node(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, dropout = DROPOUT, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
	prev_layer = 'rnn'+str(layer)

model.add_node(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, dropout = DROPOUT, return_sequences = False), name='rnn'+str(LAYERS), input = prev_layer)
model.add_node(Dropout(DROPOUT), name = 'final' , input = 'rnn'+str(LAYERS))
model.add_node(LSTMhdecoder(DIM,HIDDEN_SIZE,V_de,HIDDEN_SIZE, enc_name = 'final', dec_input_name = 'shifted_de'), name = 'dec', inputs = ['final','shifted_de'], merge_mode = 'join_dec')
model.add_node(SplitDecoder(0), name = 'dec1', input = 'dec')
model.add_node(SplitDecoder(1), name = 'dec2', input = 'dec')
model.add_node(SplitDecoder(2), name = 'dec3', input = 'dec')
model.add_node(SplitDecoder(3), name = 'dec4', input = 'dec')

model.add_output(name = 'output_1', input = 'dec1')
model.add_output(name = 'output_2', input = 'dec2')
model.add_output(name = 'output_3', input = 'dec3')
model.add_output(name = 'output_4', input = 'dec4')
optimizer = RMSprop(clipnorm = CLIP)
model.compile(loss = { 'output_1' : 'categorical_crossentropy', 'output_2' : 'categorical_crossentropy', 'output_3' : 'categorical_crossentropy', 'output_4' : 'categorical_crossentropy'}, optimizer= optimizer)

pat = 0
train_history = {'loss' : [], 'val_meteor' : []}
best_val_meteor = 0

print("saving stuff...")
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)

print("training model with {} parameters...".format(model.get_n_params()))
NB = len(b_X_tr)
for iteration in xrange(EPOCH):
	print('_' * 50)

	train_history['loss'] += [0]
	for j in xrange(NB):
		[X_tr] = b_X_tr[j]
		[Y_tr_1, Y_tr_2, Y_tr_3, Y_tr_4, Y_tr_shifted] = b_Y_tr[j]
		print('iteration {}/{} bucket {}/{}'.format(iteration+1,EPOCH, j+1,NB))

		eh = model.fit({'input_en' : X_tr , 'input_de' : Y_tr_shifted ,  'output_1' : Y_tr_1, 'output_2' : Y_tr_2, 'output_3' : Y_tr_3, 'output_4' : Y_tr_4}, batch_size = BATCH_SIZE, nb_epoch = 1, verbose = True)

		for key in ['loss']:
			train_history[key][-1] += eh.history[key][0]

	predicted = model.predict({'input_en' : X_val, 'input_de' : Y_val_shifted}, batch_size = BATCH_SIZE)
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
		pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))
	if pat == PATIENCE:
		break
print("DONE.".format(model.get_n_params()))
pickle.dump({'dicts' : dicts, 'train_history' : train_history, 'hierarchical' : HIERARCHICAL},open(PREFIX + FOOTPRINT + '.meta', 'w'))

