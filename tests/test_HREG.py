import os,multiprocessing as mp,sys
sys.path.append(os.path.abspath("."))
import numpy as np
from HREG import HREG

class_name = "HREG"
ext = f".{class_name.lower()}"
def test_randvec()->None:
    out = open(f"out({class_name}).csv", "w")
    dataset=f"randvec_{class_name}"   
    K = 100
    ef= 200
    for i in range(10):
        line = f"Acc{i}/%,"
        exps = [3,4,5]
        for exp in exps:
            dir_path = os.path.join(dataset,f"10^{exp}")
            for f in os.listdir(dir_path):
                hreg : HREG = HREG()
                if(f.endswith(ext)):
                    q = np.random.rand(128)
                    print("======================================")
                    model_path = os.path.join(dir_path,f)
                    hreg.load(model_path)
                    res = hreg.kNN_search(q,K,ef)
                    real = hreg.real_kNN(q,K)
                    
                    score = 0
                    for i in range(K):
                        if res[i] in real:
                            score +=1
                    print(f"Recall: {score*100//K}%")
                    line += f"{score*100//K},"

        line = line.removesuffix(",") + '\n'
        out.write(line)

def build_from_file(path:str):
    exp = int(path[-5])
    M = 2**exp

    dir_name = os.path.dirname(path)
    hnsw_path = path.replace(".npy",ext)
    
    if(os.path.basename(hnsw_path) not in os.listdir(dir_name)):
        hreg = HREG()
        hreg.build(path,M)
        hreg.save(hnsw_path)

def build_for_all():
    dataset=f"randvec_{class_name}"   
    for dir_name in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_name)
        for f in os.listdir(dir_path):
            if(f.endswith(".npy")):
                file_path = os.path.join(dir_path,f)
                if (f.replace(".npy",ext) not in os.listdir(dir_path)):
                    build_from_file(file_path)

        
dataset_dir=f"randvec_{class_name}" 
def clean():
      
    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir,dir_name)
        
        for f in os.listdir(dir_path):
            if(f.endswith(ext)):
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)


if __name__ == "__main__":
    # clean()
    # build_for_all()
    test_randvec()
    pass
