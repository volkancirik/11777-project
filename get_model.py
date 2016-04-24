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

def dan(RNN, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, DROPMODALITY = False):
	'''
	DAN with fc of a CNN
	'''
	from keras.layers.averagelayer import Average

	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_input(name = 'input_de', input_shape = (None,), dtype = 'int64')

	model.add_node(Layer(),name = 'shifted_de', input = 'input_de')
	model.add_node(Dense(HIDDEN_SIZE,activation = 'relu'), name = 'context_img', input = 'input_img')
	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')
	model.add_node(Average(), name = 'avg_en0', input = 'embedding')

	prev_layer = 'avg_en0'
	for layer in xrange(LAYERS -1):
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'avg_en'+str(layer+1), input = prev_layer)
		model.add_node(Dropout(DROPOUT), name = 'avg_en'+str(layer+1)+'_d' , input = 'avg_en'+str(layer+1))
		prev_layer = 'avg_en'+str(layer+1)+'_d'

	if DROPMODALITY:
		model.add_node(DropModality([HIDDEN_SIZE,HIDDEN_SIZE]), name = 'dm', inputs = [prev_layer,'context_img'], merge_mode = 'concat', concat_axis = 1)
		model.add_node(Dense(HIDDEN_SIZE,activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'merged', input = 'dm')
	else:
		model.add_node(Dense(HIDDEN_SIZE,activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'merged', inputs = [prev_layer, 'context_img'], merge_mode = 'concat',  concat_axis = -1)
	model.add_node(Dropout(DROPOUT), name = 'merged_d' , input = 'merged')

	return model, 'merged_d'


def model_0(RNN, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, DROPMODALITY = False):
	'''
	Enc-dec with fc of a CNN
	'''
	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_input(name = 'input_de', input_shape = (None,), dtype = 'int64')

	model.add_node(Layer(),name = 'shifted_de', input = 'input_de')
	model.add_node(Dense(HIDDEN_SIZE,activation = 'relu'), name = 'context_img', input = 'input_img')
	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

	prev_layer = 'embedding'
	for layer in xrange(LAYERS -1):
		model.add_node(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, dropout = DROPOUT, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
		prev_layer = 'rnn'+str(layer)

	model.add_node(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, dropout = DROPOUT, return_sequences = False), name='rnn'+str(LAYERS), input = prev_layer)

	if DROPMODALITY:
		model.add_node(DropModality([HIDDEN_SIZE,HIDDEN_SIZE]), name = 'dm', inputs = ['rnn'+str(LAYERS),'context_img'], merge_mode = 'concat', concat_axis = 1)
		model.add_node(Dense(HIDDEN_SIZE,activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'merged', input = 'dm')
	else:
		model.add_node(Dense(HIDDEN_SIZE, activation = 'relu', W_regularizer = l1l2(l1 = 0.00001, l2 = 0.00001)), name = 'merged', inputs = ['rnn'+str(LAYERS), 'context_img'], merge_mode = 'concat')
	model.add_node(Dropout(DROPOUT), name = 'merged_d' , input = 'merged')
	prev_layer = 'merged_d'

	return model, prev_layer


def model_1(RNN, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, DROPMODALITY = False):
	'''
	Attention for english and using fc of a CNN
	'''
	model = Graph()

	model.add_input(name = 'input_img', input_shape = (IMG_SIZE,))
	model.add_input(name = 'input_en', input_shape = (None,), dtype = 'int64')
	model.add_input(name = 'input_de', input_shape = (None,), dtype = 'int64')

	model.add_node(Layer(),name = 'shifted_de', input = 'input_de')

	model.add_node(Dense(HIDDEN_SIZE,activation = 'relu'), name = 'context_img', input = 'input_img')
	model.add_node(Embedding(V_en,HIDDEN_SIZE,mask_zero=True),name = 'embedding', input = 'input_en')

	prev_layer = 'embedding'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, dropout = DROPOUT, return_sequences = True), name = 'rnn'+str(layer), input = prev_layer)
		prev_layer = 'rnn'+str(layer)

	return model, prev_layer

def get_model(model_id, unit, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, dropmodality = False, use_hierarchical = False):
	print('building model...')
	RNN = UNIT[unit]
	models = { 1 : model_1, 0 : model_0, -1 : dan}

	model,prev_layer =  models[model_id](RNN,IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, LAYERS, DROPOUT, DROPMODALITY = dropmodality)
	DIM = int(pow(V_de,0.25))

	if model_id == 1:

		model.add_node(LSTMAttentionDecoder(DIM,HIDDEN_SIZE,V_de,HIDDEN_SIZE, enc_name = prev_layer, dec_input_name = 'shifted_de', img_name = 'context_img'), name = 'dec', inputs = [prev_layer, 'shifted_de', 'context_img'], merge_mode = 'join_att_dec')

	else:
		model.add_node(LSTMhdecoder(DIM,HIDDEN_SIZE,V_de,HIDDEN_SIZE, enc_name = prev_layer, dec_input_name = 'shifted_de'), name = 'dec', inputs = [prev_layer,'shifted_de'], merge_mode = 'join_dec')
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

	return model
