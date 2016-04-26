from __future__ import division
import numpy as np
from scipy.io import loadmat


def read_align(align_path, rounded=True):
    alignments = np.genfromtxt(align_path, dtype=None, delimiter=' ')
    if rounded:        
        temp = np.copy(alignments)
        for i, a in enumerate(alignments):
            # interval is [s, f)
            s = int(round(a[0] / 1000))
            f = int(round(a[1] / 1000))
            temp[i] = (s, f, a[2])
        alignments = temp
    return alignments


def get_segments(hogs_path, alignments):
    hogs = loadmat(hogs_path)['hogs']
    segments = []    
    for a in alignments:
        segments.append(hogs[a[0]:a[1],])
    segments = np.array(segments)
    return segments


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        alignments = read_align(sys.argv[1], rounded=True)        
        segments = get_segments(sys.argv[2], alignments)        