import sys,os,matplotlib.pyplot as plt
sys.path.append(os.path.abspath('.'))
from HKNN import HKNN,LapHKNN
from UPHKNN import UPHKNN,ExpUPHKNN,LapUPHKNN
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import ndcg_score
test_class = LapHKNN
class_name = test_class.__name__
ext = f".{class_name.lower()}"
K = 100
K_query = 100
ef_query = 200

def test_DP(dataset,exp,test):
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")
    
    return_res = {}
    if f"result_{test}.pkl" not in os.listdir(model_dir):
        
        dir_path = os.path.join(model_dir,test)
        for f in os.listdir(dir_path):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(dir_path,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))
                avg_rec = 0
                avg_pw_dist_mean = 0
                avg_pw_dist_max = 0
                avg_ndcg = 0
                for q in tqdm(test_data):
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

                        print(f"Mean Pairwise Distance: {pw_dist_mean}")
                        print(f"Max Pairwise Distance: {pw_dist_max}")

                        # pw_cos_sim_res = cosine_similarity(q.reshape(1, -1),data_res)
                        # pw_cos_sim_real = cosine_similarity(q.reshape(1, -1),data_real)
                        # ndcg = ndcg_score(pw_cos_sim_real,pw_cos_sim_res)
                        # avg_ndcg += ndcg

                        # print(f"NDCG: {ndcg}")



                avg_rec /= test_data.shape[0]
                avg_pw_dist_mean /= test_data.shape[0]
                avg_pw_dist_max /= test_data.shape[0]
                avg_ndcg /= test_data.shape[0]

                if test=="epsilon":
                    return_res[h.epsilon] = (avg_rec,avg_pw_dist_mean,avg_pw_dist_max,avg_ndcg)
        
        return_res = dict(sorted(return_res.items()))
        pickle.dump(return_res,open(os.path.join(model_dir,f"result_{test}.pkl"),"wb"))
    else:
        return_res = pickle.load(open(os.path.join(model_dir,f"result_{test}.pkl"),"rb"))
    return return_res

def build_DP_model(vecfile_path,model_path,**kwargs):
    exp = int(vecfile_path[-5])
    epsilon = 1
    M = int(1.5*2**exp)
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
        var_range = [1,3,5,7,9]
    elif test=="M":
        var_range = [8,16,24,32,64]
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

def test_base(dataset,exp,test):
    model_dir=os.path.join(f"{dataset}_{class_name}",f"10^{exp}")   
    return_res = None
    if f"result_{test}.pkl" not in os.listdir(model_dir):
        for f in os.listdir(model_dir):
            h = test_class()
            if f.endswith(ext):
                model_path = os.path.join(model_dir,f)
                h.load(model_path)
                test_data = np.load(os.path.join(f"{dataset}","test.npy"))

                avg_rec = 0
                avg_pw_dist_mean = 0
                avg_pw_dist_max = 0
                avg_ndcg = 0
                for q in tqdm(test_data):
                    print("======================================")
                    res,_ = h.kNN_search(q,K_query,ef_query)
                    real,_ = h.real_kNN(q,K)
                    
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
                    print(f"Mean Pairwise Distance: {pw_dist_mean}")
                    print(f"Max Pairwise Distance: {pw_dist_max}")
                    
                    # pw_cos_sim_res = cosine_similarity(q.reshape(1, -1),data_res)
                    # pw_cos_sim_real = cosine_similarity(q.reshape(1, -1),data_real)
                    # ndcg = ndcg_score(pw_cos_sim_real,pw_cos_sim_res)
                    # avg_ndcg += ndcg
                    # print(f"NDCG: {ndcg}")

                avg_rec /= test_data.shape[0]
                avg_pw_dist_mean /= test_data.shape[0]
                avg_pw_dist_max /= test_data.shape[0]
                avg_ndcg /= test_data.shape[0]
        return_res = (avg_rec,avg_pw_dist_mean,avg_pw_dist_max,avg_ndcg)
        pickle.dump(return_res,open(os.path.join(model_dir,f"result_{test}.pkl"),"wb"))
    else:
        return_res = pickle.load(open(os.path.join(model_dir,f"result_{test}.pkl"),"rb"))
    return return_res

def build_base_model(vecfile_path,model_path):
    exp = int(vecfile_path[-5])
    M = int(1.5*2**exp)
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
    for dataset in ["randvec","DEEP","GloVe"]:
        for exp in [3,4]:
            test = "epsilon"
            test_class = HKNN
            class_name= test_class.__name__
            ext = f".{class_name.lower()}"
            datasets_dir=f"{dataset}_{class_name}"
            build_base_from_file(dataset,exp)
            avg_rec_HKNN, avg_pw_dist_mean_HKNN, avg_pw_dist_max_HKNN, avg_ndcg_HKNN = test_base(dataset,exp,test)

            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
            ax1.set_ylim([0,1.1])
            
            ax1.axhline(avg_rec_HKNN,label=class_name,color="black",linestyle="--")
            ax2.axhline(avg_pw_dist_mean_HKNN,label=class_name,color="black",linestyle="--")
            ax3.axhline(avg_pw_dist_max_HKNN,label=class_name,color="black",linestyle="--")

            for tc in [LapHKNN,ExpUPHKNN,LapUPHKNN]:
                test_class = tc
                class_name= test_class.__name__
                ext = f".{class_name.lower()}"
                datasets_dir=f"{dataset}_{class_name}"
                build_DP_test_from_file(dataset,exp,test)
                res = test_DP(dataset,exp,test)

                epsilons = res.keys()
                avg_recs = [res[epsilon][0] for epsilon in epsilons]
                avg_pw_dist_mean = [res[epsilon][1] for epsilon in epsilons]
                avg_pw_dist_max = [res[epsilon][2] for epsilon in epsilons]
                avg_ndcg = [res[epsilon][3] for epsilon in epsilons]
                
                ax1.plot(epsilons,avg_recs,label=class_name.removesuffix("HKNN"),marker="o")
                ax2.plot(epsilons,avg_pw_dist_mean,label=class_name.removesuffix("HKNN"),marker="o")
                ax3.plot(epsilons,avg_pw_dist_max,label=class_name.removesuffix("HKNN"),marker="o")

                ax1.set_xlabel(test)
                ax2.set_xlabel(test)
                ax3.set_xlabel(test)
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax1.set_ylabel("Recall")
                ax2.set_ylabel("Mean Pairwise Distance")
                ax3.set_ylabel("Max Pairwise Distance")

           
            fig.suptitle(f"{dataset} 10^{exp}")
            
            fig.savefig(f"{dataset}_10^{exp}.png")
            plt.close()
