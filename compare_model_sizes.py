# -*- coding: utf-8 -*-
from __future__ import print_function

from get_model import get_model
"""
# of parameters comparison for different models
"""

BASE_N_PARAMETER =  1690792 # 1932204 # 1690792
unit = 'gru'
print("enc-dec with {} : # of parameters of {} with {} layers and {} units".format(unit,BASE_N_PARAMETER,2,128))

HIERARCHICAL = True
IMG_SIZE = 4096
MAXLEN = 40 # 113 # 40
V_en = 10085 # 11967 # 10085
V_de = 8688 # 14106 # 8688
prec = 0.01
for MODEL in [0,1]:
	m_param = 0
	for unit in ['gru','lstm']:
		for layer in [1,2,3]:
			h_min = 0
			h_max = 256
			HIDDEN_SIZE = 0
			m_param = 0
			while not (BASE_N_PARAMETER - BASE_N_PARAMETER*prec <= m_param <=  BASE_N_PARAMETER + BASE_N_PARAMETER*prec):
				print("-->",BASE_N_PARAMETER - BASE_N_PARAMETER*prec, m_param, BASE_N_PARAMETER + BASE_N_PARAMETER*prec, MODEL, unit, layer, HIDDEN_SIZE)
				if m_param >= BASE_N_PARAMETER:
					h_max = HIDDEN_SIZE
				else:
					h_min = HIDDEN_SIZE
				HIDDEN_SIZE = (h_min + h_max)/2
				m_param = get_model(MODEL, unit, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, layer, 0 , use_hierarchical = HIERARCHICAL).get_n_params()

			m_param = get_model(MODEL, unit, IMG_SIZE, MAXLEN, V_en, V_de, HIDDEN_SIZE, layer, 0, use_hierarchical = HIERARCHICAL).get_n_params()
			print("# of parameters of the model {} use : --unit {} --layers {} --hidden {} --model {}".format(m_param,unit,layer,HIDDEN_SIZE,MODEL))
