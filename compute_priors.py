import numpy as np
import os
import glob
import sys
import binvox_rw
from collections import defaultdict
# Store mapping from directory hash to class type, can get from the .json?
HASH_TO_CLASS = {}
PRIOR_DIR = "priors/"


"""
Returns a dict that look like:
{
    category: [<absolute .binvox filepaths for the category>].
    ...
}
"""


# we will reuse this in the dataloader
def get_category_filepaths(dataset_path):
    d = defaultdict(list)

    for c in [f.path for f in os.scandir(dataset_path) if f.is_dir()]:
        d[os.path.basename(c)] = list(
            glob.glob('{}/**/*.binvox'.format(c), recursive=True))

    return d


"""
Returns a dict that look like:
{
    category: <average of shapes>
    ...
}
"""


def compute_priors(category_filepaths):
    d = dict()
    for c in category_filepaths:
        data = np.zeros((32,32,32))
        for f in category_filepaths[c]:
            m1 = binvox_rw.read_as_3d_array(open(f, 'rb'))
            data += np.transpose(m1.data, (0, 2, 1)).astype(np.float32)

        d[c] = data / len(category_filepaths[c])

    return d


"""
Iterate through category priors and save them as .npy files. 

Save as <category name>.npy.

Use HASH_TO_CLASS here. 
"""


def save_priors(category_priors):
    for c in category_priors:
        np.save('{}.npy'.format(c), category_priors[c])


if __name__ == "__main__":
    # usage compute_priors.py <dataset_path>
    # Use some argument parser for this ^
    # Store result in dataset_path
    if not os.path.exists(PRIOR_DIR):
        os.mkdir(PRIOR_DIR)

    category_model_fps = get_category_filepaths(sys.argv[1])

    category_priors = compute_priors(category_model_fps)

    save_priors(category_priors)
