import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.abspath('.'))
from HNSW import HNSW,LapUPHNSW
from HKNN import HKNN
from UPHKNN import UPHKNN,ExpUPHKNN,LapUPHKNN
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import pairwise_distances

test_class = LapUPHKNN
class_name = test_class.__name__
ext = f".{class_name.lower()}"
K = 100
K_query = 100
ef_query = 200

def test_DP(dataset,exp,test):
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")
    data_dir = os.path.join(f"{dataset}",f"10^{exp}")
    return_res = {}
    if not os.path.exists(os.path.join(model_dir,f"result_{test}.pkl")):
        dir_path = os.path.join(model_dir,test)
        for f in os.listdir(dir_path):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(dir_path,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))
                avg_rec = 0
                avg_acc = 0
                avg_pw_dist_mean = 0
                avg_pw_dist_max = 0
                avg_q_dist_mean = 0
                avg_q_dist_max = 0

                labeled = False
                if os.path.exists(os.path.join(data_dir,f"y_{dataset}_10^{exp}.npy")):
                    y_data = np.load(os.path.join(data_dir,f"y_{dataset}_10^{exp}.npy"))
                    labeled = True
                if os.path.exists(os.path.join(f"{dataset}","y_test.npy")):
                    y_test = np.load(os.path.join(f"{dataset}","y_test.npy"))

                for i,q in enumerate(tqdm(test_data)):
                        print("======================================")
                        res,dist_res = h.kNN_search(q,K_query,ef_query)
                        real,dist_real = h.real_kNN(q,K)
                        
                        TP = len(set(res)&set(real))
                        
                        rec = TP/float(K)
                        avg_rec += rec

                        print(f"Recall: {rec}")

                        data_res = h.data[res]
                        data_real = h.data[real]
                        pw_dist = pairwise_distances(data_res,data_real,n_jobs=-1)
                        pw_dist_mean = np.mean(pw_dist)
                        pw_dist_max = np.max(pw_dist)
                        avg_pw_dist_mean += pw_dist_mean
                        avg_pw_dist_max += pw_dist_max

                        avg_q_dist_mean += np.mean(dist_res)
                        avg_q_dist_max += np.max(dist_res)

                        if labeled:
                            acc = np.sum(y_data[res]==y_test[i])/K_query
                            avg_acc += acc

                avg_rec /= test_data.shape[0]
                avg_pw_dist_mean /= test_data.shape[0]
                avg_pw_dist_max /= test_data.shape[0]
                avg_acc /= test_data.shape[0]
                avg_q_dist_mean /= test_data.shape[0]
                avg_q_dist_max /= test_data.shape[0]

                res = (avg_rec,avg_acc,avg_pw_dist_mean,avg_pw_dist_max,avg_q_dist_mean,avg_q_dist_max)
                if test=="epsilon":
                    return_res[h.epsilon] = res
                elif test=="M":
                    return_res[h.M] = res
        return_res = dict(sorted(return_res.items()))
        pickle.dump(return_res,open(os.path.join(model_dir,f"result_{test}.pkl"),"wb"))
    else:
        return_res = pickle.load(open(os.path.join(model_dir,f"result_{test}.pkl"),"rb"))
    return return_res

def build_DP_model(vecfile_path,model_path,**kwargs):
    exp = int(vecfile_path[-5])
    epsilon = 1
    M = 25
    unprotected = 0.5

    for key, val in kwargs.items():
        if key=="epsilon":
            epsilon = val
        elif key=="M":
            M = val
        elif key=="unprotected":
            unprotected = val
    
    if  test_class in [UPHKNN,ExpUPHKNN,LapUPHKNN]:
        h = test_class(epsilon=epsilon,path=vecfile_path,num_unprotected=unprotected)
    else:
        h = test_class(epsilon=epsilon,path=vecfile_path)
    h.build(M)
    h.save(model_path)    

def build_DP_test_from_file(dataset,exp,test):
    vecfile_path = os.path.join(f"{dataset}",f"10^{exp}",f"{dataset}_10^{exp}.npy")
    root_dir = f"{dataset}_{class_name}"
    if root_dir not in os.listdir(os.curdir):
        os.mkdir(root_dir)

    exp_dir = os.path.join(root_dir,f"10^{exp}")
    if f"10^{exp}" not in os.listdir(root_dir):
        os.mkdir(exp_dir)

    test_dir = os.path.join(exp_dir,test)
    if(test not in os.listdir(exp_dir)):
        os.mkdir(test_dir)
    file_name = os.path.basename(vecfile_path)

    if test=="epsilon":
        var_range = range(1,10)
    elif test=="M":
        var_range = range(20,50,5)
    else:
        exit(1)
    
    for var in var_range:
        info=f"{test}={var}_"
        test_path = os.path.join(exp_dir,test)
        h_path = os.path.join(exp_dir,test,info+file_name.replace(".npy",ext))
        
        if(os.path.basename(h_path) not in os.listdir(test_path)):
            if test=="epsilon":
                build_DP_model(vecfile_path,h_path,epsilon=var)
            elif test=="M":
                build_DP_model(vecfile_path,h_path,M=var)
            else:
                exit(1)

def clean_test_res():
    for rt in os.listdir(os.curdir):
        if os.path.isdir(rt):
            for dir in os.listdir(rt):
                dir_path = os.path.join(rt,dir)
                if os.path.isdir(dir_path) and dir.startswith("10^"):
                    for f in os.listdir(dir_path):
                        if f.endswith(".pkl"):
                            os.remove(os.path.join(dir_path,f))

def test_base(dataset,exp):
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")  
    data_dir=os.path.join(f"{dataset}",f"10^{exp}") 
    return_res = None
    if not os.path.exists(os.path.join(model_dir,f"result.pkl")):
        for f in os.listdir(model_dir):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(model_dir,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))

                avg_rec = 0
                avg_acc = 0
                avg_pw_dist_mean = 0
                avg_pw_dist_max = 0
                avg_q_dist_mean = 0
                avg_q_dist_max = 0

                labeled = False
                if os.path.exists(os.path.join(data_dir,f"y_{dataset}_10^{exp}.npy")):
                    y_data = np.load(os.path.join(data_dir,f"y_{dataset}_10^{exp}.npy"))
                    labeled = True
                if os.path.exists(os.path.join(f"{dataset}","y_test.npy")):
                    y_test = np.load(os.path.join(f"{dataset}","y_test.npy"))

                for i,q in enumerate(tqdm(test_data)):
                    print("======================================")
                    res,dist_res = h.kNN_search(q,K_query,ef_query)
                    real,dist_real = h.real_kNN(q,K)
                    
                    TP = len(set(res)&set(real))
                    rec = TP/float(K)
                    avg_rec += rec
                    print(f"Recall: {rec}")
                    data_res = h.data[res]
                    data_real = h.data[real]
                    pw_dist = pairwise_distances(data_res,data_real,n_jobs=-1)
                    pw_dist_mean = np.mean(pw_dist)
                    pw_dist_max = np.max(pw_dist)
                    avg_pw_dist_mean += pw_dist_mean
                    avg_pw_dist_max += pw_dist_max
                    avg_q_dist_mean += np.mean(dist_res)
                    avg_q_dist_max += np.max(dist_res)

                    if labeled:
                        acc = np.sum(y_data[res]==y_test[i])/K_query
                        avg_acc += acc
                    
                avg_rec /= test_data.shape[0]
                avg_acc /= test_data.shape[0]
                avg_pw_dist_mean /= test_data.shape[0]
                avg_pw_dist_max /= test_data.shape[0]
                avg_q_dist_mean /= test_data.shape[0]
                avg_q_dist_max /= test_data.shape[0]
        return_res = (avg_rec,avg_acc,avg_pw_dist_mean,avg_pw_dist_max,avg_q_dist_mean,avg_q_dist_max)
        pickle.dump(return_res,open(os.path.join(model_dir,f"result.pkl"),"wb"))
    else:
        return_res = pickle.load(open(os.path.join(model_dir,f"result.pkl"),"rb"))
    return return_res

def build_base_model(vecfile_path,model_path):
    exp = int(vecfile_path[-5])
    M = 25
    h = test_class(path=vecfile_path)
    h.build(M)
    h.save(model_path)

def build_base_from_file(dataset,exp):
    vecfile_path = os.path.join(f"{dataset}",f"10^{exp}",f"{dataset}_10^{exp}.npy")
    if not os.path.exists(vecfile_path):
        return
    root_dir = f"{dataset}_{class_name}"
    if root_dir not in os.listdir(os.curdir):
        os.mkdir(root_dir)

    exp_dir = os.path.join(root_dir,f"10^{exp}")
    if f"10^{exp}" not in os.listdir(root_dir):
        os.mkdir(exp_dir)
    file_name = os.path.basename(vecfile_path)

    h_path = os.path.join(exp_dir,file_name.replace(".npy",ext))
    if(os.path.basename(h_path) not in os.listdir(exp_dir)):
        build_base_model(vecfile_path,h_path)


if __name__ == "__main__":
    # clean_test_res()
    if not os.path.exists("figure"):
        os.mkdir("figure")
    for exp in [4]:
        for test in ["M","epsilon"]:
            if test=="M":
                test_label = "K"
            elif test=="epsilon":
                test_label = r"$\epsilon$"
            for dataset in ["randvec","DEEP","GloVe","MNIST","SIFT"]:
                fig0,ax0 = plt.subplots(1,1)
                fig1,ax1 = plt.subplots(1,1)
                fig2,ax2 = plt.subplots(1,1)
                fig3,ax3 = plt.subplots(1,1)
                fig4,ax4 = plt.subplots(1,1)
                fig5,ax5 = plt.subplots(1,1)
                axs = [ax0,ax1,ax2,ax3,ax4,ax5]
                figs = [fig0,fig1,fig2,fig3,fig4,fig5]
                y_labels = ["Recall","Accuracy","APD","MPD","AQD","MQD"]
                ax0.set_ylim([0,1.1])
                ax1.set_ylim([0,1.1])

                colors =  ["red","black"]
                for it,test_class in enumerate([HKNN,HNSW]):
                    class_name= test_class.__name__
                    ext = f".{class_name.lower()}"
                    datasets_dir=f"{dataset}_{class_name}"
                    build_base_from_file(dataset,exp)
                    avg_res = test_base(dataset,exp)
                    
                    for i,ax in enumerate(axs):
                        ax.axhline(avg_res[i],label=class_name,color=colors[it],linestyle="--")

                for test_class in [ExpUPHKNN,LapUPHKNN,LapUPHNSW]:
                    
                    class_name= test_class.__name__
                    ext = f".{class_name.lower()}"
                    datasets_dir=f"{dataset}_{class_name}"
                    build_DP_test_from_file(dataset,exp,test)
                    res = test_DP(dataset,exp,test)

                    keys = res.keys()

                    for i, ax in enumerate(axs):
                        avg_res = [res[key][i] for key in keys]
                        ax.plot(keys,avg_res,label=class_name,marker="o")
                        ax.set_xlabel(test_label)
                        ax.legend()
                        ax.set_ylabel(y_labels[i])

                for i,fig in enumerate(figs):
                    fig.savefig(os.path.join("figure",f"{dataset}_{y_labels[i]}_{test}.png"),bbox_inches="tight")
                    plt.close()
