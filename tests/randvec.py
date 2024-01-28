import numpy as np
import os

def gen_randvec_file(*exp):
    data_dir = "randvec_DPHREG"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)   
    dim = 128
    for i in exp:
        dir_path = os.path.join(data_dir,f"10^{i}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = f"randvec{dim}_10^{i}.npy"
        if filename in dir_path:
            continue
        filepath : str = os.path.join(dir_path, filename)
        f = open(filepath, "w")
        size = 10**i
        rand_vec = np.random.rand(size,dim)
        np.save(filepath,rand_vec)
        f.close()
    
if __name__ == "__main__":
    gen_randvec_file(3,4,5,6)