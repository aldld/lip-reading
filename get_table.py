from __future__ import division
from tabulate import tabulate
import os, sys
import numpy as np
import operator


def get_tabulated_results(exp_dir, subset=False):
    headers = ['Number of components', 'WER', 'WER Std. Dev.','IOU', 'IOU Std. Dev.']
    if subset:
        headers = ['Number of components', '%Training data', 'WER', 'WER Std. Dev.','IOU', 'IOU Std. Dev.']

    table = []
    for exp in os.listdir(exp_dir):        
        exp_path = os.path.join(exp_dir, exp)
        
        # Assuming name format 'Ntest_Mn.npy'
        n = exp.split('.')[0].split('_')[1][:-1]        
        if subset:
            perc = exp[:2]
       
        dist, iou = np.load(exp_path).tolist()        
        dist_mean = np.mean(reduce(operator.add, dist.values())) / 75
        dist_std = np.std(reduce(operator.add, dist.values())) / 75
        iou_mean = np.mean(reduce(operator.add, iou.values()))
        iou_std = np.std(reduce(operator.add, iou.values()))

        dist_mean = '%.4f' % dist_mean
        dist_std = '%.4f' % dist_std
        iou_mean = '%.4f' % iou_mean
        iou_std = '%.4f' % iou_std

        if not subset:
            table.append([n, dist_mean, dist_std, iou_mean, iou_std])
        else:
             table.append([n, perc, dist_mean, dist_std, iou_mean, iou_std])
    table = sorted(table, key=lambda r: int(r[0]))
    if subset:
        table = sorted(table, key=lambda r: (int(r[0]),int(r[1])))
    print tabulate(table, headers, tablefmt='latex')

exp_dir = sys.argv[1]

get_tabulated_results(exp_dir)
