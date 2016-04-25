import sys
import cPickle as pickle
import numpy as np

'''
test a MMMT model
'''
def load_model(path):
	meta_dict = pickle.load(open(path))
	return meta_dict
meta_dict = load_model(sys.argv[1]+'.meta')
if meta_dict['train_history']['val_meteor'] != []:
	print sys.argv[1],':',np.max(meta_dict['train_history']['val_meteor'])
