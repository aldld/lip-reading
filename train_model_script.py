from models import hsmm
import sys
import importlib
utils = importlib.import_module("lip-extraction.utils") # Import with "-"

verbose = False

if len(sys.argv) == 1:
    data_dir = "/Users/eric/Programming/prog_crs/lip-reading/data/grid"
else:
    data_dir = sys.argv[1]

data = utils.get_data(data_dir, hog_flatten=True, speakers=["s4"], verbose=verbose)

hsmm.train_hsmm(
    data,
    len(utils.vocab_mapping),
    n_components=6,
    pkl_param="model.pkl",
    verbose=verbose,
    parallel=False
    )
