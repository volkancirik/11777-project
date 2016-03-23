from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, MaskedLayer, Merge, Layer
from keras.layers import recurrent
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop

from keras.layers.attention import DenseAttention, TimeDistributedAttention
UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}
CLIP = 5

def model_0(RNN, IMG_SIZE, MAXLEN, V_en, HIDDEN_SIZE, LAYERS, DROPOUT):
	'''
	Enc-dec with fc of a CNN
	'''
	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_node(Dense(HIDDEN_SIZE,activation = 'relu'), name = 'context_img', input = 'input_img') ###

	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

	prev_layer = 'embedding'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'rnn'+str(layer)+'_d', input = 'rnn'+str(layer))
		prev_layer = 'rnn'+str(layer)+'_d'

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = False), name='rnn'+str(LAYERS), input = prev_layer)
	model.add_node(RepeatVector(MAXLEN), name='rv_en', input='rnn'+str(LAYERS))
#	model.add_node(RepeatVector(MAXLEN), name='rv_img', input='input_img')
	model.add_node(RepeatVector(MAXLEN), name='rv_img', input='context_img')

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='decoder_rnn0', inputs = ['rv_en', 'rv_img'], merge_mode = 'concat',  concat_axis = -1)
	model.add_node(Dropout(DROPOUT),name = 'decoder_rnn0_d', input = 'decoder_rnn0')

	prev_layer = 'decoder_rnn0'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'decoder_rnn' + str(layer+1), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'decoder_rnn'+str(layer+1)+'_d', input = 'decoder_rnn' + str(layer+1))
		prev_layer = 'decoder_rnn'+str(layer+1)+'_d'

	return model, prev_layer


def model_1(RNN, IMG_SIZE, MAXLEN, V_en, HIDDEN_SIZE, LAYERS, DROPOUT):
	'''
	Attention for english and using fc of a CNN
	'''
	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_node(Dense(HIDDEN_SIZE,activation = 'relu'), name = 'context_img', input = 'input_img') ###
	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

	prev_layer = 'embedding'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'rnn'+str(layer)+'_d', input = 'rnn'+str(layer))
		prev_layer = 'rnn'+str(layer)+'_d'

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='encoder_context', input = prev_layer)
#	model.add_node(RepeatVector(MAXLEN), name='recurrent_context', input='input_img')
	model.add_node(RepeatVector(MAXLEN), name='recurrent_context', input='context_img')

	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True), name='attention', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att') ##
	model.add_node(Dropout(DROPOUT),name = 'attention_d', input = 'attention')

	prev_layer = 'attention_d'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'decoder_rnn' + str(layer+1), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'decoder_rnn'+str(layer+1)+'_d', input = 'decoder_rnn' + str(layer+1))
		prev_layer = 'decoder_rnn'+str(layer+1)+'_d'

	return model,prev_layer

def model_2(RNN, IMG_SIZE, MAXLEN, V_en, HIDDEN_SIZE, LAYERS, DROPOUT):
	'''
	Attention for english and conv feature maps
	'''

	model = Graph()
	IMG_SIZE = 196
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_input(name = 'input_img', input_shape = (None,IMG_SIZE))
	model.add_node(Layer(),name = 'img', input = 'input_img')
	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

	prev_layer = 'embedding'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'rnn'+str(layer)+'_d', input = 'rnn'+str(layer))
		prev_layer = 'rnn'+str(layer)+'_d'

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='encoder_context', input = prev_layer)
	model.add_node(RNN(HIDDEN_SIZE), name='rnn', input = prev_layer)
	model.add_node(RepeatVector(MAXLEN), name='recurrent_context', input='rnn')

	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True), name='attention_en', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')
	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True, enc_name = 'img'), name='attention_img', inputs=['img','recurrent_context'], merge_mode = 'join_att')

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'attention', inputs = ['attention_en', 'attention_img'], merge_mode = 'concat',  concat_axis = -1)
	model.add_node(Dropout(DROPOUT),name = 'attention_d', input = 'attention')

	prev_layer = 'attention_d'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'decoder_rnn' + str(layer+1), input = prev_layer)
		model.add_node(Dropout(DROPOUT),name = 'decoder_rnn'+str(layer+1)+'_d', input = 'decoder_rnn' + str(layer+1))
		prev_layer = 'decoder_rnn'+str(layer+1)+'_d'

	return model

def get_model(model_id, unit, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, use_hierarchical = False):
	print('building model...')
	RNN = UNIT[unit]
	models = { 2 : model_2, 1 : model_1, 0 : model_0}
	model,prev_layer =  models[model_id](RNN,IMG_SIZE, MAXLEN, V_en, HIDDEN_SIZE, LAYERS, DROPOUT)

	if use_hierarchical:
		DIM = int(pow(V_de,0.25)) + 1
		model.add_node(TimeDistributedDense(DIM), name='tdd_1', input= prev_layer)
		model.add_node(TimeDistributedDense(DIM), name='tdd_2', input= prev_layer)
		model.add_node(TimeDistributedDense(DIM), name='tdd_3', input= prev_layer)
		model.add_node(TimeDistributedDense(DIM), name='tdd_4', input= prev_layer)

		model.add_node(Activation('softmax'), name = 'softmax_1',input = 'tdd_1')
		model.add_node(Activation('softmax'), name = 'softmax_2',input = 'tdd_2')
		model.add_node(Activation('softmax'), name = 'softmax_3',input = 'tdd_3')
		model.add_node(Activation('softmax'), name = 'softmax_4',input = 'tdd_4')

		model.add_output(name='output_1', input='softmax_1')
		model.add_output(name='output_2', input='softmax_2')
		model.add_output(name='output_3', input='softmax_3')
		model.add_output(name='output_4', input='softmax_4')

		optimizer = RMSprop(clipnorm = CLIP)
		model.compile(optimizer = optimizer, loss = {'output_1': 'categorical_crossentropy','output_2': 'categorical_crossentropy','output_3': 'categorical_crossentropy','output_4': 'categorical_crossentropy'})
	else:
		model.add_node(TimeDistributedDense(V_de), name='tdd', input= prev_layer)
		model.add_node(Activation('softmax'), name = 'softmax',input = 'tdd')
		model.add_output(name='output', input='softmax')
		optimizer = RMSprop(clipnorm = CLIP)
		model.compile(optimizer = optimizer, loss = {'output': 'categorical_crossentropy'})
	return model
