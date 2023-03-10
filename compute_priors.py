import os

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
def get_category_filepaths(dataset_path): # we will reuse this in the dataloader
    raise NotImplementedError


"""
Returns a dict that look like:
{
    category: <average of shapes>
    ...
}
""" 
def compute_priors(category_filepaths):
    raise NotImplementedError



"""
Iterate through category priors and save them as .npz files. 

Save as <category name>.npz.

Use HASH_TO_CLASS here. 
"""
def save_priors(category_priors):
    raise NotImplementedError




if __name__ == "__main__":
    # usage compute_priors.py -d <dataset_path>
    # Use some argument parser for this ^
    # Store result in dataset_path    
    if not os.path.exists(PRIOR_DIR):
        os.mkdir(PRIOR_DIR)        

    category_model_fps = get_category_filepaths(dataset_path)

    category_priors = compute_priors(category_model_fps)

    save_priors(category_priors)
    



