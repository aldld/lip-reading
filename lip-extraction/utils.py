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

def get_segments(hogs_path, alignments, flatten=False):
    hogs = loadmat(hogs_path)['hogs']
    segments = []    
    for a in alignments:
        segment_hogs = hogs[a[0]:a[1],]
        if flatten:
            segment_hogs = segment_hogs.reshape((segment_hogs.shape[0], -1))
        segments.append()
    segments = np.array(segments)    
    return segments

def get_vocab_mapping():
    d = {0: 'sil', 1: 'bin', 2: 'lay', 3: 'place', 4: 'set', 5: 'blue', 6: 'green', 
        7: 'red', 8: 'white', 9: 'at', 10: 'by', 11: 'in', 12: 'with', 13: 'a', 14: 'b', 
        15: 'c', 16: 'd', 17: 'e', 18: 'f', 19: 'g', 20: 'h', 21: 'i', 22: 'j', 23: 'k', 
        24: 'l', 25: 'm', 26: 'n', 27: 'o', 28: 'p', 29: 'q', 30: 'r', 31: 's', 32: 't', 
        33: 'u', 34: 'v', 35: 'x', 36: 'y', 37: 'z', 38: '0', 39: '1', 40: '2', 41: '3',
        42: '4', 43: '5', 44: '6', 45: '7', 46: '8', 47: '9', 48: 'again', 49: 'now', 
        50: 'please', 51: 'soon'}

    return d

def get_word_frame_nums(data_dir, file_out=None):
    word_frame_nums = defaultdict(list)

    for speaker_dir in data_dir:
        align_dir = os.path.join(speaker_dir, 'align')
        for align_file in os.listdir(align_dir):
            alignments = read_align(os.path.join(align_dir, align_file))
            for a in alignments:    
                word_frame_nums[a[2]].append(a[1]-a[0])

    if file_out:
        np.save(file_out, word_frame_nums)

    return word_frame_nums

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        alignments = read_align(sys.argv[1], rounded=True)        
        segments = get_segments(sys.argv[2], alignments)   
        import pdb; pdb.set_trace()     