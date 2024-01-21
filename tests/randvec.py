import numpy as np
import os,time,shutil

test_classes = ["HGraph","HNSW","HNSWH","JLHNSW","HREG","HMST"]

def gen_randvec_file(*exp):
    if not os.path.exists(f"randvec"):
        os.makedirs(f"randvec")   
    dim = 128
    for i in exp:
        dir_path = os.path.join("randvec",f"10^{i}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = f"randvec{dim}_10^{i}.npy"
        if filename in dir_path:
            continue
        filepath : str = os.path.join(dir_path, filename)
        f = open(filepath, "w")
        size = 10**i
        rand_vec = np.random.randint(-1024,1023,size=(size,dim))
        np.save(filepath,rand_vec)
        f.close()
    
if __name__ == "__main__":
    gen_randvec_file(3,4,5,6)