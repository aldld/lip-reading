from __future__ import division
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
import os
import cPickle as pickle

vocab_mapping = {0: 'sil', 1: 'bin', 2: 'lay', 3: 'place', 4: 'set', 5: 'blue', 6: 'green', 
    7: 'red', 8: 'white', 9: 'at', 10: 'by', 11: 'in', 12: 'with', 13: 'a', 14: 'b', 
    15: 'c', 16: 'd', 17: 'e', 18: 'f', 19: 'g', 20: 'h', 21: 'i', 22: 'j', 23: 'k', 
    24: 'l', 25: 'm', 26: 'n', 27: 'o', 28: 'p', 29: 'q', 30: 'r', 31: 's', 32: 't', 
    33: 'u', 34: 'v', 35: 'x', 36: 'y', 37: 'z', 38: 'zero', 39: 'one', 40: 'two', 41: 'three',
    42: 'four', 43: 'five', 44: 'six', 45: 'seven', 46: 'eight', 47: 'nine', 48: 'again', 49: 'now', 
    50: 'please', 51: 'soon', 52: 'sp', 53: 'sil2'}
vocab_mapping_r = {word: i for i, word in vocab_mapping.items()} # Reverse mapping

entire_set = {'s%s'%i for i in xrange(1,35) if i!=21}
test_set = {'s%s'%i for i in xrange(28,35)}
train_set = entire_set - test_set
entire_set, test_set, train_set = list(entire_set), list(test_set), list(train_set)


def read_align(align_path, rounded=True):
    """ Returns the alignemnt data in the given .align file. """
    alignments = np.genfromtxt(align_path, dtype=None, delimiter=' ')
    if rounded:        
        temp = np.copy(alignments)
        for i, a in enumerate(alignments):
            # interval is [s, f)
            s = int(round(a[0] / 1000))
            f = int(round(a[1] / 1000))
            temp[i] = (s, f, a[2])
        alignments = temp
    # This will make a second silence token to be used as a sink state
    # in our hsmm transition matrix 
    alignments[-1][2] = vocab_mapping[53] 
    return alignments


def get_word_frame_nums(data_dir, file_out=None):
    """
    Traverses align folders of all speakers under the given data_dir 
    and returns observed durations for each observed word.
    """
    word_frame_nums = defaultdict(list)

    for speaker_dir in os.listdir(data_dir):
        align_dir = os.path.join(data_dir, speaker_dir, 'align')
        if not os.path.exists(align_dir):
            print 'align directory %s does not exist' % align_dir
            continue
        for align_file in os.listdir(align_dir):
            alignments = read_align(os.path.join(align_dir, align_file))
            for a in alignments:    
                word_frame_nums[a[2]].append(a[1]-a[0])

    if file_out:
        np.save(file_out, word_frame_nums)

    return word_frame_nums


def get_chain(hog_path, align_path, speaker, hog_fn, hog_flatten=False, return_gt=False):
    """ 
    Returns the state, observation chain corresponding to given hog and align files.
    """
    hogs = loadmat(hog_path)['hogs']
    if not hogs.size:
        if not return_gt:           
            return None
        else:
            return None, None
    alignments = read_align(align_path)
    chain = defaultdict(list)

    chain['speaker'] = speaker
    chain['hog_fn'] = hog_fn

    if return_gt:
        gt = []

    for a in alignments:
        observed_hogs = hogs[a[0]:a[1],]
        
        # This is mainly for 'sp' observations that are less than a frame long
        if observed_hogs.size == 0:
                continue
        
        if hog_flatten:
            observed_hogs = observed_hogs.reshape((observed_hogs.shape[0], -1))

        chain['state_seq'].append(vocab_mapping_r[a[2]])
        chain['obs'].append(observed_hogs)

        if return_gt:
            gt.extend([vocab_mapping_r[a[2]]] * (a[1]-a[0]))
    
    if not return_gt:
        return chain
    else:
        return chain, gt

def get_data(data_dir, hog_flatten=False, speakers=None, verbose=True):
    """
    For each speaker under data_dir, combines and returns chains from corresponding
    hog and align files.
    """
    data = []

    for speaker_id in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_id)

        if (not os.path.isdir(speaker_path)) or (speakers is not None and speaker_id not in speakers):
            continue

        align_dir = os.path.join(speaker_path, 'align')
        if not os.path.exists(align_dir):
            print 'align directory %s does not exist' % align_dir
            continue
        hog_dir = os.path.join(speaker_path, 'hog')
        if not os.path.exists(hog_dir):
            print 'hog directory %s does not exist' % hog_dir
            continue
        for hog_file in os.listdir(hog_dir):
            if not hog_file.endswith('.mat'):
                continue
            if verbose:
                print hog_file
            align_file = hog_file.split('.')[0] + '.align'
            hog_path = os.path.join(hog_dir, hog_file)
            align_path = os.path.join(align_dir, align_file)
            chain = get_chain(hog_path, align_path, speaker_id, hog_file, hog_flatten)
            if not chain:
                continue
            data.append(chain)

    np.random.shuffle(data)
    return data


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        chain = get_chain(sys.argv[1], sys.argv[2])   
        import pdb; pdb.set_trace()
    else:
        data_dir = 'C:\\Users\\Berkay Antmen\\Desktop\\412Final\\data'
        # print get_word_frame_nums(data_dir).keys()
        print get_data(data_dir)
