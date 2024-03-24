import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from LapHREG import LapHREG
from ExpHREG import ExpHREG
from LapExpHREG import LapExpHREG
import numpy as np
from tqdm import tqdm
import pickle

test_class = LapExpHREG
class_name = test_class.__name__
ext = f".{class_name.lower()}"

def test_dataset(dataset,exp)->None:
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")
    test = "epsilon"
    K = 100
    ef= 200
    avgs = {}
    if f"{test}.pkl" not in os.listdir(model_dir):
        dir_path = os.path.join(model_dir,test)
        for f in os.listdir(dir_path):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(dir_path,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))
                avg = 0
                for q in tqdm(test_data):
                        print("======================================")
                        res = h.kNN_search(q,K,ef)
                        real = h.real_kNN(q,K)
                        
                        score = len(set(res)&set(real))
                        
                        rec = score/float(K)
                        avg += rec
                        print(f"Recall: {rec}")
                avg /= test_data.shape[0]
                if test=="epsilon":
                    avgs[h.epsilon] = avg
        
        avgs = dict(sorted(avgs.items()))
        pickle.dump(avgs,open(os.path.join(model_dir,f"{test}.pkl"),"wb"))
    else:
        avgs = pickle.load(open(os.path.join(model_dir,f"{test}.pkl"),"rb"))
    # plt.ylabel("Recall")
    # plt.xlabel(test)
    # plt.ylim(0,1.1)
    # plt.plot(avgs.keys(),avgs.values())
    # plt.legend()
    # plt.savefig(os.path.join(model_dir,f"{test}.png"))
    # plt.close()
    return avgs

def build_model(vecfile_path,model_path,epsilon=0.2):
    exp = int(vecfile_path[-5])
    M = 2**exp
    
    h = test_class(epsilon=epsilon,path=vecfile_path)
    h.build(M)
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

    for epsilon in [0.5,1,2,3,5]:
        priv_budget_info=f"{test_name}={epsilon}_"
        test_path = os.path.join(exp_dir,test_name)
        h_path = os.path.join(exp_dir,test_name,priv_budget_info+file_name.replace(".npy",ext))
        
        if(os.path.basename(h_path) not in os.listdir(test_path)):
            build_model(vecfile_path,h_path,epsilon)

if __name__ == "__main__":
    for dataset in ["randvec","MNIST","GloVe"]:
        for exp in [3,4]:
            plt.ylim(0,1.1)
            for tc in [LapHREG,ExpHREG,LapExpHREG]:
                test_class = tc
                class_name= test_class.__name__
                ext = f".{class_name.lower()}"
                datasets_dir=f"{dataset}_{class_name}"
                build_from_file(dataset,exp)
                avgs = test_dataset(dataset,exp)
                plt.plot(avgs.keys(),avgs.values(),label=class_name.removesuffix("HREG"))
                
            plt.xlabel("epsilon")
            plt.ylabel("Recall")
            plt.title(f"{dataset} 10^{exp}")
            plt.legend()
            plt.savefig(f"{dataset}_10^{exp}.png")
            plt.close()
