from __future__ import division
import cPickle as pickle
from models import hsmm
import numpy as np
from nltk.metrics.distance import edit_distance
import importlib
utils = importlib.import_module("lip-extraction.utils") # Import with "-"
import sys, os
from collections import defaultdict


EXP_DIR = 'experiments_hsmm'

def inf_to_voc(inf):
    return [vocab_mapping[i] for i in inf]


def load_model(model_path):
    model = pickle.load(open(model_path))
    model['vocab_size'] = 53
    model_h = hsmm.build_hsmm(**model)
    return model_h


# Return the data to be fed for evaluation and the 
# respective ground truth
def get_eval_data(hog_path, align_path):
    chain, gt = utils.get_chain(hog_path, align_path, None, None, None, True)
    if chain:
        d = np.vstack(tuple([chain['obs'][i] for i in xrange(len(chain['obs']))]))
        return d, gt
    else:
        return np.empty(0), None


def model_evaluate(model, d, gt):
    model_h.add_data(d, trunc=25)
    inf = model_h.states_list[0].stateseq
    
    inf = list(inf)    
    dist = edit_distance(gt, inf)
    
    s_gt, s_inf = set(gt), set(inf)
    iou = len(s_gt.intersection(s_inf)) / len(s_gt.union(s_inf))

    return dist, iou


def evaluate(data_dir, test_set, model):
    dist_results = defaultdict(list)
    iou_results = defaultdict(list)
    for speaker in test_set:
        print 'Current speaker ', speaker
        speaker_path = os.path.join(data_dir, speaker)
        hog_dir = os.path.join(speaker_path, 'hog')
        for hog in os.listdir(hog_dir):
            if not hog.endswith('.mat'):
                continue
            hog_path = os.path.join(hog_dir, hog)
            align = hog.split('.')[0] + '.align'
            align_path = os.path.join(speaker_path, 'align', align)
            d, gt = get_eval_data(hog_path, align_path)
            if not d.size:
                continue
            dist, iou = model_evaluate(model, d, gt)
            dist_results[speaker].append(dist)
            iou_results[speaker].append(iou)
    return dist_results, iou_results


data_dir = '/ais/gobi4/freud/temp/data'
#test_set = [utils.test_set[-1]]
test_set = utils.test_set
#test_set = utils.train_set
model_path = sys.argv[1]
model_h = load_model(model_path)

print 'test set: ', test_set
print 'model: ', model_path

exp_name = sys.argv[2]

experiment_path = os.path.join(EXP_DIR, exp_name)

print 'experiment path: ', experiment_path

dist_results, iou_results = evaluate(data_dir, test_set, model_h)
results = [dist_results, iou_results]

np.save(experiment_path, results)
