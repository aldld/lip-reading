from models import hsmm

import importlib
utils = importlib.import_module("lip-extraction.utils") # Import with "-"

data_dir = "/Users/eric/Programming/prog_crs/lip-reading/data/grid"

data = utils.get_data(data_dir, hog_flatten=True, speakers=["s4"])
print data
exit()

hsmm.train_hsmm(
    data,
    len(utils.vocab_mapping),
    n_components=6,
    pkl_param="model.pkl",
    verbose=True
    )
