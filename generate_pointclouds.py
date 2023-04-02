import os
import glob
import numpy as np

SIZE=128
PC_DIR="pointclouds/"
PRIOR_DIR = "priors/"
DELTA=1

def surface(x):
    false_vals = np.zeros(x.shape, dtype=bool)
    false_vals[0,:,:] = True
    false_vals[x.shape[0]-1,:,:] = True
    false_vals[:,0,:] = True
    false_vals[:,x.shape[1]-1,:] = True
    false_vals[:,:,0] = True
    false_vals[:,:,x.shape[2]-1] = True
    for i in range(3):
        false_vals |= np.roll(x, shift=-1, axis=i) == False
        false_vals |= np.roll(x, shift=1, axis=i) == False

    return x & false_vals



if __name__ == "__main__":
    # usage generate_pointclouds.py <dataset_path>
    # Use some argument parser for this ^
    # Store result in dataset_path
    if not os.path.exists(PC_DIR):
        os.mkdir(PC_DIR)
    
    for f in os.scandir(PRIOR_DIR):
        data = np.load(f.path)
        data = data > 0.3 # Threshhold
        pc = surface(data)
        s = pc.sum()
        if s == 0:
            continue
        points_per_voxel = 2048 // s
        points_per_voxel_end = 2048 - points_per_voxel * s
        point_cloud = [0] * 2048
        for i, (x, y, z) in enumerate(zip(*np.where(pc))):
            num = points_per_voxel + 1 if i < points_per_voxel_end else points_per_voxel
            shift = 0 if i < points_per_voxel_end else (points_per_voxel_end) * (points_per_voxel + 1)
            idx = i if i < points_per_voxel_end else i - points_per_voxel_end
	    
            for p in range(num):
                point_cloud[shift + idx * num + p] = np.array([x, y, z]) + np.random.rand(1,3) / DELTA
        
        point_cloud = np.concatenate(point_cloud)

        np.save('{}/{}_pointcloud.npy'.format(PC_DIR, os.path.basename(f.path).split('.')[0]), point_cloud)
 
