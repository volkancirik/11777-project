from __future__ import print_function

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

	word_idx = meta_dict['word_idx']
	idx_word = meta_dict['idx_word']

	with open(arch) as json_file:
		architecture = json.load(json_file)
	model = model_from_json(architecture)
	model.load_weights(model_filename)

	return model, word_idx, idx_word

### get arguments
parser = get_parser_nmt_test()
p = parser.parse_args()

PATH = p.path
MINI_BATCH = True
BATCH_SIZE = 128
model, word_idx, idx_word = load_model(PATH)

X_test, _ , X_test_img = prepare_test(mini_batch = MINI_BATCH)

if MINI_BATCH:
	predicted = model.predict({'input_en' : X_test, 'input_img' : X_test_img}, batch_size=BATCH_SIZE)
else:
	raise NotImplementedError("sgd not implemented!")

SAMPLES = 5
V = predicted['output'].shape[2]
vocab = range(V)
S = []
for i in xrange(predicted['output'].shape[0]):

	EOS = word_idx['</s>']
	greedy = [ np.argmax(p) for p in predicted['output'][i]]
	try:
		greedy_clipped = greedy[:greedy.index(EOS)]
	except:
		greedy_clipped = greedy
		pass

	seq = [ idx_word[idx] for idx in greedy_clipped]
	seq_p = np.sum([ np.log(predicted['output'][i,j,idx]) for j,idx in enumerate(greedy_clipped)])
	s = {'sentence' : " ".join(seq), 'prob' : seq_p}
	samples = [s]

	for k in xrange(SAMPLES):
		s = {'sentence' : [], 'prob' : 0}
		seq = []
		seq_p = 0
		for j in xrange(predicted['output'].shape[1]):
			prob = predicted['output'][i,j,:].reshape((V,))
			token_idx = np.random.choice(vocab, p = prob)
			token = idx_word[token_idx]
			if token == '</s>':
				break
			seq += [token]
			seq_p += np.log(predicted['output'][i,j,token_idx])
		s['sentence'] = " ".join(seq)
		s['prob'] = np.sum(seq_p)
		samples += [s]
	S += [samples]
for i in xrange(2):
	for j in xrange(SAMPLES+1):
		print("{} ::: {}".format(S[i][j]['prob'],S[i][j]['sentence']))
	print(50*"_")
