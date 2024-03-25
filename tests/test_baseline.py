import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from HREG import HREG
from HNSW import HNSW
import numpy as np
from tqdm.auto import tqdm
test_class = HREG
class_name = test_class.__name__
ext = f".{class_name.lower()}"

def test_dataset(dataset,exp)->None:
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")   
    K = 100
    ef= 200
    avg = 0
    for f in os.listdir(model_dir):
        h = test_class()
        if f.endswith(ext):
            model_path = os.path.join(model_dir,f)
            h.load(model_path)
            test_data = np.load(os.path.join(f"{dataset}","test.npy"))

            avg = 0
            for q in tqdm(test_data):
                print("======================================")
                res = h.kNN_search(q,K,ef)
                real = h.real_kNN(q,K)
                
                score = len(set(res)&set(real))
                
                rec = score/float(K)
                print(f"Recall: {rec}")
                avg+=rec

            avg /= test_data.shape[0]

    plt.ylabel("Recall")
    plt.ylim(0,1.1)
    plt.axhline(avg)
    plt.legend()
    plt.savefig(os.path.join(model_dir,f"avg_rec.png"))
    plt.close()

def build_model(vecfile_path,model_path):
    exp = int(vecfile_path[-5])
    M = 2**exp
    h = test_class(path=vecfile_path)
    h.build(M)
    h.save(model_path)

def build_from_file(dataset,exp):
    vecfile_path = os.path.join(f"{dataset}",f"10^{exp}",f"{dataset}_10^{exp}.npy")
    root_dir = f"{dataset}_{class_name}"
    if root_dir not in os.listdir(os.curdir):
        os.mkdir(root_dir)

    exp_dir = os.path.join(root_dir,f"10^{exp}")
    if f"10^{exp}" not in os.listdir(root_dir):
        os.mkdir(exp_dir)
    file_name = os.path.basename(vecfile_path)

    h_path = os.path.join(exp_dir,file_name.replace(".npy",ext))
    if(os.path.basename(h_path) not in os.listdir(os.path.dirname(vecfile_path))):
        build_model(vecfile_path,h_path)

if __name__ == "__main__":
    # clean_test_res()
    for dataset in ["randvec","MNIST","GloVe","DEEP"]:
        for exp in [3,4]:
            for tc in [HREG,HNSW]:
                test_class = tc
                class_name= test_class.__name__
                ext = f".{class_name.lower()}"
                datasets_dir=f"{dataset}_{class_name}"
                build_from_file(dataset,exp)
                test_dataset(dataset,exp)
