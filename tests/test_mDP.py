import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from LapHREG import LapHREG
from ExpHREG import ExpHREG
import numpy as np
from joblib import Parallel,delayed
from tqdm import tqdm

test_class = LapHREG
class_name = test_class.__name__
ext = f".{class_name.lower()}"

def query_test(q,h,K=100,ef=200):
    print("======================================")
    res = h.kNN_search(q,K,ef)
    real = h.real_kNN(q,K)
    
    score = len(set(res)&set(real))
    
    rec = score/float(K)
    print(f"Recall: {rec}")
    return rec

def test_dataset(dataset,exp)->None:
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")   
    K = 100
    ef= 200
    tests = ["epsilon"]
    for test in tests:
        dir_path = os.path.join(model_dir,test)
        avgs = {}
        # avgs_q = {}
        for f in os.listdir(dir_path):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(dir_path,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))

                res = Parallel(n_jobs=-1)(delayed(query_test)(q,h,K,ef) for q in tqdm(test_data))
                avg = np.mean(res)
                if test=="epsilon":
                    if not h.quantize:
                        avgs[h.epsilon] = avg
                    # else:
                    #     avgs_q[h.epsilon] = avg
        
        avgs = dict(sorted(avgs.items()))
        # avgs_q = dict(sorted(avgs_q.items()))

        plt.ylabel("Recall")
        plt.xlabel(test)
        plt.ylim(0,1.1)
        plt.plot(avgs.keys(),avgs.values(),label="not quantized")
        # plt.plot(avgs_q.keys(),avgs_q.values(),label="quantized")
        plt.legend()
        plt.savefig(os.path.join(model_dir,f"{test}.png"))
        plt.close()

def build_model(vecfile_path,model_path,epsilon=0.2,quantize=False):
    exp = int(vecfile_path[-5])
    M = 2**exp
    
    h = test_class(epsilon=epsilon,path=vecfile_path)
    h.build(M,quantize=quantize)
    h.save(model_path)

def build_from_file(dataset,exp):
    test_name = "epsilon"
    vecfile_path = os.path.join(f"{dataset}",f"10^{exp}",f"{dataset}_10^{exp}.npy")
    root_dir = f"{dataset}_{class_name}"
    if root_dir not in os.listdir(os.curdir):
        os.mkdir(root_dir)

    exp_dir = os.path.join(root_dir,f"10^{exp}")
    if f"10^{exp}" not in os.listdir(root_dir):
        os.mkdir(exp_dir)

    test_dir = os.path.join(exp_dir,test_name)
    if(test_name not in os.listdir(exp_dir)):
        os.mkdir(test_dir)
    file_name = os.path.basename(vecfile_path)

    for epsilon in [0.1,0.2,0.5,1,2,5]:
        priv_budget_info=f"{test_name}={epsilon}_"
        test_path = os.path.join(exp_dir,test_name)
        h_path = os.path.join(exp_dir,test_name,priv_budget_info+file_name.replace(".npy",ext))
        # hq_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".npy",f"_q{ext}"))
        
        if(os.path.basename(h_path) not in os.listdir(test_path)):
            build_model(vecfile_path,h_path,epsilon)
        # if(os.path.basename(hq_path) not in os.listdir(test_path)):
        #     build_model(vecfile_path,hq_path,epsilon,quantize=True)

if __name__ == "__main__":
    for dataset in ["randvec","MNIST","CIFAR10"]:
        datasets_dir=f"{dataset}_{class_name}"
        for exp in [3,4,5]:
            build_from_file(dataset,exp)
            test_dataset(dataset,exp)
