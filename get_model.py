from keras.models import Graph
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector, Dense, Dropout, Merge, Layer
from keras.layers import recurrent
from keras.optimizers import RMSprop

from keras.layers.attention import DenseAttention, TimeDistributedAttention
UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}

def model_0(RNN, IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE):
	'''
	Enc-dec with fc of a CNN
	'''
	## TODO : LAYER
	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,DIM))

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = False), name='rnn', input='input_en')
	model.add_node(RepeatVector(MAXLEN), name='rv_en', input='rnn')
	model.add_node(RepeatVector(MAXLEN), name='rv_img', input='input_img')

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='decoder', inputs = ['rv_en', 'rv_img'], merge_mode = 'concat',  concat_axis = -1)

	model.add_node(TimeDistributedDense(V), name='tdd', input= 'decoder')
	model.add_node(Activation('softmax'), name = 'softmax',input = 'tdd')
	model.add_output(name='output', input='softmax')
	optimizer = RMSprop(clipnorm = 5)
	print("compiling model...")
	model.compile(optimizer = optimizer, loss = {'output': 'categorical_crossentropy'})
	return model


def model_1(RNN, IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE):
	'''
	Attention for english and using fc of a CNN
	'''
	## TODO : LAYER
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
	return model


def model_2(RNN, IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE):
	'''
	Attention for english and conv feature maps
	'''
	## TODO : LAYER
	model = Graph()
	IMG_SIZE = 196
	model.add_input(name = 'input_en', input_shape = (None,DIM))
	model.add_input(name = 'input_img', input_shape = (None,IMG_SIZE))
	model.add_node(Layer(),name = 'img', input = 'input_img')

	model.add_node(RNN(HIDDEN_SIZE), name='rnn', input='input_en')
	model.add_node(RepeatVector(MAXLEN), name='recurrent_context', input='rnn')
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='encoder_context', input='input_en')

	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True), name='attention_en', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')
	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True, enc_name = 'img'), name='attention_img', inputs=['img','recurrent_context'], merge_mode = 'join_att')

	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name = 'decoder', inputs = ['attention_en', 'attention_img'], merge_mode = 'concat',  concat_axis = -1)

	model.add_node(TimeDistributedDense(V), name='tdd', input= 'decoder')
	model.add_node(Activation('softmax'), name = 'softmax',input = 'tdd')
	model.add_output(name='output', input='softmax')
	optimizer = RMSprop(clipnorm = 5)
	print("compiling model...")
	model.compile(optimizer = optimizer, loss = {'output': 'categorical_crossentropy'})
	return model


def get_model(model_id, unit, IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE):
	print('building model...')
	RNN = UNIT[unit]
	models = { 2 : model_2, 1 : model_1, 0 : model_0}
	return models[model_id](RNN,IMG_SIZE, DIM, MAXLEN, V, HIDDEN_SIZE)
