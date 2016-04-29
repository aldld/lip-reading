import cPickle as pickle
from models import hsmm
import numpy as np
import importlib
utils = importlib.import_module("lip-extraction.utils") # Import with "-"
import sys

model = pickle.load(open(sys.argv[1]))

model['vocab_size'] = 53

model_h = hsmm.build_hsmm(**model)

chain, gt = utils.get_chain('/ais/gobi4/freud/temp/data/s4/hog/bgbh2s.mat', '/ais/gobi4/freud/temp/data/s4/align/bgbh2s.align', 's4', None, 'bgbh2', True)
d = np.vstack(tuple([chain['obs'][i] for i in xrange(len(chain['obs']))]))
print d.shape

#model_h.add_data(d, trunc=30)

chain, gt = utils.get_chain('/ais/gobi4/freud/temp/data/s4/hog/bgbh3p.mat', '/ais/gobi4/freud/temp/data/s4/align/bgbh3p.align', 's4', None, 'bgbh3p', True)
d = np.vstack(tuple([chain['obs'][i] for i in xrange(len(chain['obs']))]))
print d.shape

model_h.add_data(d, trunc=30)

#np.seterr(all='raise')

#model_h.resample_model()

print (model_h.states_list[0].stateseq)
print gt
#print utils.inf_to_voc(model_h.states_list[0].stateseq)

#print (model_h.states_list[1].stateseq)
#print utils.inf_to_voc(model_h.states_list[1].stateseq)
