from __future__ import division
import numpy as np

def read_align(path, rounded=True):
    alignments = np.genfromtxt(path, dtype=None, delimiter=' ')
    if rounded:        
        temp = np.copy(alignments)
        for i, a in enumerate(alignments):
            # interval is [s, f)
            s = int(round(a[0] / 1000))
            f = int(round(a[1] / 1000))
            temp[i] = (s, f, a[2])
        alignments = temp
    return alignments

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        print read_align(sys.argv[1], rounded=True)
        print read_align(sys.argv[1], rounded=False)