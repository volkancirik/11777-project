from __future__ import print_function

import theano
from keras.models import model_from_json

import json, time, datetime, os, sys
import cPickle as pickle
import numpy as np

from prepare_data import prepare_test
from utils import get_parser_nmt_test
'''
test a MMMT model
'''
def load_model(path):
	meta = path + '.meta'
	arch = path + '.arch'
	model_filename = path + '.model'

	meta_dict = pickle.load(open(meta))

	dicts = meta_dict['dicts']
	use_hierarchical = meta_dict['hierarchical']

	with open(arch) as json_file:
		architecture = json.load(json_file)
	model = model_from_json(architecture)
	model.load_weights(model_filename)

	return model, dicts, use_hierarchical

def get_table(predicted, gold, length, SAMPLES = 10, HIERARCHICAL = True):

	V_de = len(dicts['word_idx_de'])
	dim = int(pow(V_de,0.25)) + 1
	vocab = range(V_de)
	S = []
	EOS = dicts['word_idx_de']['</s>']
	delta = [0]*len(gold)

	for i in xrange(SAMPLES):
		if HIERARCHICAL:
			probabilities = []
			for j in xrange(predicted['output_1'].shape[1]):
				prob_token = np.zeros((V_de,1))
				for idx in range(V_de):
					idx_1 = idx / (dim**3)
					idx_2 = (idx - (dim**3)*idx_1) / (dim**2)
					idx_3 = (idx - (dim**3)*idx_1 - (dim**2)*idx_2) / (dim)
					idx_4 = idx % (dim)

					prob_token[idx] = predicted['output_1'][i,j,idx_1] * predicted['output_2'][i,j,idx_2] * predicted['output_3'][i,j,idx_3] * predicted['output_4'][i,j,idx_4]
				probabilities += [ prob_token/ np.sum(prob_token)]
		else:
			probabilities = predicted['output'][i]

		gold_probs = [ probabilities[j][dicts['word_idx_de'][word]] for j,word in enumerate(gold[i].split())]
		greedy = [ np.argmax(p) for p in probabilities ]
		greedy_probs = [ np.max(p) for p in probabilities ]
		try:
			greedy_clipped = greedy[:greedy.index(EOS)]
		except:
			greedy_clipped = greedy
			pass

		seq = [ dicts['idx_word_de'][idx] for idx in greedy_clipped]
		seq_p = np.sum([ np.log(p) for p in greedy_probs[:len(greedy_clipped)]])
		gold_p = np.sum([ np.log(p) for p in gold_probs])

		delta[i] = gold_p - seq_p
	return delta


def decode_predicted(source, gold, predicted, dicts, SAMPLES = 10, HIERARCHICAL = True, random_samples = 0, func = "eval"):

	V_de = len(dicts['word_idx_de'])
	dim = int(pow(V_de,0.25)) + 1
	vocab = range(V_de)
	S = []
	EOS = dicts['word_idx_de']['</s>']

	for i in xrange(SAMPLES):
		if HIERARCHICAL:
			probabilities = []
			for j in xrange(predicted['output_1'].shape[1]):
				prob_token = np.zeros((V_de,1))
				for idx in range(V_de):
					idx_1 = idx / (dim**3)
					idx_2 = (idx - (dim**3)*idx_1) / (dim**2)
					idx_3 = (idx - (dim**3)*idx_1 - (dim**2)*idx_2) / (dim)
					idx_4 = idx % (dim)

					prob_token[idx] = predicted['output_1'][i,j,idx_1] * predicted['output_2'][i,j,idx_2] * predicted['output_3'][i,j,idx_3] * predicted['output_4'][i,j,idx_4]
				probabilities += [ prob_token/ np.sum(prob_token)]
		else:
			probabilities = predicted['output'][i]

		greedy = [ np.argmax(p) for p in probabilities]
		greedy_probs = [ np.max(p) for p in probabilities]
		try:
			greedy_clipped = greedy[:greedy.index(EOS)]
		except:
			greedy_clipped = greedy
			pass

		seq = [ dicts['idx_word_de'][idx] for idx in greedy_clipped]
		seq_p = np.sum([ np.log(p) for p in greedy_probs[:len(greedy_clipped)]])
		s = {'sentence' : " ".join(seq), 'prob' : seq_p}
		samples = [s]

		for k in xrange(random_samples):
			s = {'sentence' : [], 'prob' : 0}
			seq = []
			seq_p = 0

			for prob_token in probabilities:
				token_idx = np.random.choice(vocab, p = prob_token.reshape(V_de,))
				token = dicts['idx_word_de'][token_idx]
				if token == '</s>':
					break
				seq += [token]
				seq_p += np.log( prob_token[token_idx])
			s['sentence'] = " ".join(seq)
			s['prob'] = np.sum(seq_p)
			samples += [s]
		S += [samples]
	for i in xrange(SAMPLES):
		for j in xrange(random_samples+1):
			if func == "test":
				print("{} ||| {} ||| {} ||| {}".format(source[i],gold[i],S[i][j]['sentence'],S[i][j]['prob']))
			else:
				print("{}".format(S[i][j]['sentence']))
		if func == "test":
			print(50*"_")

def get_sentences(path_prefix = '../data/', test_source = 'val.en', test_target = 'val.de', suffix = '.task1'):
	gold = []
	source = []
	for line_g,line_s in zip(open(path_prefix + test_target + suffix),open(path_prefix + test_source + suffix)):
		gold += [line_g.strip()]
		source += [line_s.strip()]
	return gold, source

### get arguments
parser = get_parser_nmt_test()
p = parser.parse_args()

FUNC = p.func
PATH = p.path
SUFFIX = p.suffix
M_TYPE = p.m_type
REPEAT= {'full' : True, 'truncated' : False, 'debug' : False, 'task1' : False}[SUFFIX] and (M_TYPE in set(["text+img","image"]))
BATCH_SIZE = 64
SAMPLES = len(X_test) if p.samples <= 0 else p.samples

model, dicts, HIERARCHICAL = load_model(PATH)
X_test, _ , X_test_img, _ = prepare_test(dicts['word_idx_en'],repeat = REPEAT, suffix = {'full' : '.all.tokenized.unkified', 'truncated' : '.truncated', 'debug' : '.debug', 'task1' : '.task1'}[SUFFIX])
gold,source = get_sentences(suffix = {'full' : '.all.tokenized.unkified', 'truncated' : '.truncated', 'debug' : '.debug', 'task1' : '.task1'}[SUFFIX])

X_test = X_test[:SAMPLES]
X_test_img = X_test_img[:SAMPLES]
if M_TYPE == 'text+image' or M_TYPE == 'image+text':
	predicted = model.predict({'input_en' : X_test, 'input_img' : X_test_img}, batch_size=BATCH_SIZE)
elif M_TYPE == "text":
	predicted = model.predict({'input_en' : X_test}, batch_size=BATCH_SIZE)
elif M_TYPE == "image":
	predicted = model.predict({'input_img' : X_test_img}, batch_size=BATCH_SIZE)
else:
	raise NotImplementedError()

if FUNC == "test" or FUNC == "eval":
	decode_predicted(source, gold, predicted, dicts, SAMPLES = SAMPLES, HIERARCHICAL = True, func = FUNC)
elif FUNC == "error":
	delta_model = get_table(predicted, gold, dicts, SAMPLES = SAMPLES, HIERARCHICAL = True)
	print(delta_model)
else:
	raise NotImplementedError()
