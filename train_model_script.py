from models import hsmm
import sys
import cPickle as pickle
import importlib
utils = importlib.import_module("lip-extraction.utils") # Import with "-"

verbose = False

# Default values
data_dir = "/Users/eric/Programming/prog_crs/lip-reading/data/grid"
n_components = 6
speakers = ['s4']

# Command line values
if len(sys.argv) >= 2:
    data_dir = sys.argv[1]
if len(sys.argv) >= 3:
    n_components = int(sys.argv[2])
if len(sys.argv) >= 4:
    percentage = int(sys.argv[3])
    num_speaker = int(float(percentage) / 100 * len(utils.train_set))
    speakers = []
    for i in xrange(num_speaker):
        speakers.append(utils.train_set[i])

print 'data dir: ', data_dir
print 'n: ', n_components
print 'num speakers: ', len(speakers)

data = utils.get_data(data_dir, hog_flatten=True, speakers=speakers, verbose=verbose)

model_str = 'model_{0}n_{1}.pkl'.format(n_components, '-'.join(speakers))

model = hsmm.train_hsmm(
        data,
        len(utils.vocab_mapping),
        n_components=n_components,
        pkl_param=model_str,
        verbose=verbose,
        parallel=False
        )
