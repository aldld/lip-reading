import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from sklearn.mixture import GMM
import pyhsmm
import pybasicbayes

import cPickle as pickle

import multiprocessing

vocab_mapping = {0: 'sil', 1: 'bin', 2: 'lay', 3: 'place', 4: 'set', 5: 'blue', 6: 'green', 
    7: 'red', 8: 'white', 9: 'at', 10: 'by', 11: 'in', 12: 'with', 13: 'a', 14: 'b', 
    15: 'c', 16: 'd', 17: 'e', 18: 'f', 19: 'g', 20: 'h', 21: 'i', 22: 'j', 23: 'k', 
    24: 'l', 25: 'm', 26: 'n', 27: 'o', 28: 'p', 29: 'q', 30: 'r', 31: 's', 32: 't', 
    33: 'u', 34: 'v', 35: 'x', 36: 'y', 37: 'z', 38: 'zero', 39: 'one', 40: 'two', 41: 'three',
    42: 'four', 43: 'five', 44: 'six', 45: 'seven', 46: 'eight', 47: 'nine', 48: 'again', 49: 'now', 
    50: 'please', 51: 'soon', 52: 'sp', 53: 'sil2'}
vocab_mapping_r = {word: i for i, word in vocab_mapping.items()} # Reverse mapping

def train_word_init_probs(data, vocab_size):
    """ Compute initial word probabilities. """
    word_init_counts = np.zeros((vocab_size,))

    for chain in data:
        if chain['state_seq'][0] in word_init_counts:
            word_init_counts[chain['state_seq'][0]] = 1
        else:
            word_init_counts[chain['state_seq'][0]] += 1

    return word_init_counts / float(vocab_size)

def train_word_trans_probs(data, vocab_size):
    """ Return the HSMM state transition matrix trained using MLE from bigram counts. """
    bigram_counts = np.zeros((vocab_size, vocab_size))

    num_bigrams_tot = 0
    for chain in data:
        for w1, w2 in zip(chain["state_seq"][:-1], chain["state_seq"][1:]):
            bigram_counts[w1, w2] += 1
        num_bigrams_tot += len(chain["state_seq"]) - 1

    return bigram_counts / float(num_bigrams_tot)

def train_word_durations(data, vocab_size):
    """ Learn lambda parameter for word durations via MLE, assuming that durations follow a Poisson distribution. """
    word_counts = np.asarray([0.0 for _ in xrange(vocab_size)])
    word_durations = np.asarray([0.0 for _ in xrange(vocab_size)])

    for chain in data:
        for idx, word in enumerate(chain['state_seq']):
            duration = len(chain['obs'][idx])

            word_counts[word] += 1
            word_durations[word] += duration

    return word_durations / word_counts

def gather_gmm_data(data, vocab_size):
    """ Returns a dictionary mapping words to their observation data matrices (of shape (num_segments, hogs_dim)) """
    # Collect segments for each word in training data.
    segments = [[] for _ in xrange(vocab_size)]
    for chain in data:
        for idx, word in enumerate(chain['state_seq']):
            segments[word].append(chain['obs'][idx])

    # Put segments for each word together, so that each word has a single data matrix.
    train_data_gmm = [None for _ in xrange(vocab_size)]
    for word in xrange(vocab_size):
        train_data_gmm[word] = np.vstack((np.asarray(seg) for seg in segments[word]))

    return train_data_gmm

def fit_gmm(pair):
    pair[0].fit(pair[1])

def train_word_gmms(train_data_gmm, n_components=6, verbose=False, parallel=False):
    """ Train word-level GMMs given the current word. """
    gmms = [GMM(n_components=n_components) for _ in train_data_gmm]

    if parallel:
        multiprocessing.Pool().map(fit_gmm, zip(gmms, train_data_gmm))
    else:
        for idx, obs in enumerate(train_data_gmm):
            if verbose:
                print "Training GMM for word %d" % idx
                #print obs.shape

            if idx == vocab_mapping_r["sil2"]:
                continue
            elif idx == vocab_mapping_r["sil"]:
                sil_obs = np.vstack((train_data_gmm[idx], train_data_gmm[vocab_mapping_r["sil2"]]))
                gmms[idx].fit(sil_obs)
            else:
                gmms[idx].fit(obs)

        gmms[vocab_mapping_r["sil2"]] = gmms[vocab_mapping_r["sil"]]

    return gmms

class FixedGaussian(pyhsmm.distributions.Gaussian):
    def resample(self, data=[]):
        return self

class FixedHSMMInitialState(pyhsmm.internals.initial_state.HSMMInitialState):
    def resample(self, data=[]):
        return self

class FixedHSMMTransitions(pyhsmm.internals.transitions.HSMMTransitions):
    def resample(self,stateseqs=[],trans_counts=None):
        return self

def build_hsmm(word_init_probs, word_trans_probs, word_dur_params, word_gmms, vocab_size, out_fn=None, eps=0.0001):
    """ Build pyhsmm HSMM from estimated parameters. """

    # Build observation GMM distributions.
    obs_distns = []
    for sk_gmm in word_gmms:
        weights = sk_gmm.weights_
        means = sk_gmm.means_
        covars = sk_gmm.covars_

        # Construct individual Gaussian mixture components.
        mix_components = []
        for i in xrange(sk_gmm.n_components):
            sigma = np.diag(covars[i, :]) # TODO: Assumes diagonal covariance.

            #mu_0 = means[i, :]


            gaussian = FixedGaussian(mu=means[i, :], sigma=sigma)

            mix_components.append(gaussian)

        obs_gmm = pybasicbayes.models.MixtureDistribution(alpha_0=1.0, components=mix_components, weights=weights)

        obs_distns.append(obs_gmm)

    # Build duration Poisson distributions.
    dur_distns = []
    for lmbda in word_dur_params:
        # Need to choose hyperparameters b/c reasons. This will constrain the prior distribution of lambda so that
        # the mean is equal to lambda and the variance is less than eps.
        beta_0 = lmbda / float(eps)
        alpha_0 = lmbda * beta_0

        poisson_duration = pyhsmm.distributions.PoissonDuration(lmbda=lmbda, alpha_0=alpha_0, beta_0=beta_0)

        dur_distns.append(poisson_duration)

    #int_state_distn = FixedHSMMInitialState(model=hsmm, pi_0=word_init_probs)
    trans_distn = FixedHSMMTransitions(num_states=len(obs_distns), alpha=1.0, trans_matrix=word_trans_probs)

    hsmm = pyhsmm.models.HSMM(alpha=1.0, obs_distns=obs_distns, dur_distns=dur_distns, trans_distn=trans_distn)
    #hsmm = pyhsmm.models.WeakLimitHDPHSMM(alpha=1.0, gamma=1.0, init_state_concentration=1.0,
    #    obs_distns=obs_distns, dur_distns=dur_distns, trans_distn=trans_distn)
    #hsmm.trans_distn.trans_matrix = word_trans_probs

    hsmm.init_state_distn = FixedHSMMInitialState(model=hsmm, pi_0=word_init_probs)
    hsmm.init_state_distn.K = vocab_size

    if out_fn is not None:
        with open(out_fn, "wb") as f:
            pickle.dump(hsmm, f)

    return hsmm

def save_params(word_init_probs, word_trans_probs, word_dur_params, word_gmms, out_file):
    params = {
        "word_init_probs": word_init_probs,
        "word_trans_probs": word_trans_probs,
        "word_dur_params": word_dur_params,
        "word_gmms": word_gmms
    }

    with open(out_file, "wb") as f:
        pickle.dump(params, f)

def train_hsmm(data, vocab_size, n_components=6, pkl_param=None, verbose=True, parallel=False):
    """ Trains a HSMM from the given complete data.

        data: List of observation sequences and state sequences.
            [
                {
                    state_seq: [word_0, word_1, word_2]
                    obs: [[frame_0, frame_1], [frame_2, frame_3, frame_4], [frame_5]],
                },
                {
                    state_seq: [word_0, word_1, word_2]
                    obs: [[frame_0, frame_1], [frame_2, frame_3, frame_4], [frame_5]],
                }
                ...
            ]
    """

    if verbose:
        print "Computing initial word probabilities..."
    word_init_probs = train_word_init_probs(data, vocab_size) # Compute initial word probabilities.
    
    if verbose:
        print "Computing word transition probabilities..."
    word_trans_probs = train_word_trans_probs(data, vocab_size) # Compute word transition probabilities.
    word_dur_params = train_word_durations(data, vocab_size) # Learn parameters of word duration distributions.

    # Gather data for training each GMM.
    if verbose:
        print "Gathering GMM training data..."
    train_data_gmm = gather_gmm_data(data, vocab_size)

    # Train word GMMs.
    if verbose:
        print "Training GMMs..."
    word_gmms = train_word_gmms(train_data_gmm, n_components=n_components, verbose=verbose, parallel=parallel)

    if pkl_param is not None:
        # Save intermediate model, if so desired.
        save_params(word_init_probs, word_trans_probs, word_dur_params, word_gmms, pkl_param)

    return build_hsmm(word_init_probs, word_trans_probs, word_dur_params, word_gmms, vocab_size)



