from randvec import gen_randvec
from NoisyDistHNSW import NoisyDistHNSW
import os,multiprocessing as mp,matplotlib.pyplot as plt,numpy as np
    
def test_randvec(exp)->None:
    out = open("out(NoisyDist).csv", "w")
    dataset=os.path.join("randvec",f"10^{exp}")   
    K = 100
    ef= 200
    test_sets = ["sigma"]
    line = "file_name,"
    for i in range(10):
        line += f"Acc{i}/%,"
    line +="Avg/%\n"
    out.write(line)
    for test_set in test_sets:
        dir_path = os.path.join(dataset,test_set)
        avgs = {}
        avgs_h = {}
        for f in os.listdir(dir_path):
            hnsw = NoisyDistHNSW()
            if(f.endswith(".hnsw") or f.endswith(".hnswh")):
                sum = 0
                line = f+','
                model_path = os.path.join(dir_path,f)
                hnsw.load(model_path)

                for t in range(10):
                    q = gen_randvec(128)
                    print("======================================")
            
                    res = hnsw.kNN_search(q,K,ef)
                    real = hnsw.real_kNN(q,K)
                    
                    score = 0
                    for i in range(K):
                        if res[i] in real:
                            score +=1
                    
                    sum +=score
                    print(f"Accuracy: {score*100//K}%")
                    line += f"{score*100//K},"

                avg = sum//10
                if test_set=="sigma":
                    if hnsw.algchoose==3:
                        avgs[hnsw.sigma] = avg
                    if hnsw.algchoose==4:
                        avgs_h[hnsw.sigma] = avg
                
                line += f"{avg}\n"
                out.write(line)
        
        avgs = dict(sorted(avgs.items()))
        avgs_h = dict(sorted(avgs_h.items()))
        sigmas = ["sigma/10","sigma/5","sigma/2","sigma","sigma*2","sigma*5","sigma*10"]
        plt.ylabel("Accuracy/%")
        plt.plot(sigmas,avgs.values(),label = "hnsw")
        plt.plot(sigmas,avgs_h.values(), label = "hnswh")
        plt.legend()
        plt.savefig(os.path.join(dataset,f"{test_set}.png"))
        plt.close()

    out.close()
        

def build_JLHNSW(vecfile_path,model_path,sigma,M,efConstr,use_heu):
    hnsw = NoisyDistHNSW()
    hnsw.set_Noise_dev(sigma)
    if(use_heu==True):
        hnsw.use_heuristic_selection()  
    hnsw.build(vecfile_path,M,efConstr)
    hnsw.save(model_path)

def build_from_file(path:str):
    scale = int(path[-5])
    M = 0
    efConstr = 100
    if scale ==3:
        M = 10
    elif scale ==4:
        M = 20
    elif scale ==5:
        M = 40
    elif scale ==6:
        M = 48

    D = np.loadtxt(path,delimiter=',',dtype=int)
    Ds = np.empty([D.shape[0],D.shape[0]],dtype=int)
    for i in range(D.shape[0]):
        sqsum = np.sum(np.square(D[i]))
        Ds[i] = np.full(D.shape[0],sqsum,dtype=int)
        
    
    R = Ds + Ds.transpose() - 2*np.matmul(D, D.transpose())

    dists = R.flatten()
    base_sigma = np.std(dists)
    sigmas = [base_sigma/10,base_sigma/5,base_sigma/2,base_sigma,base_sigma*2,base_sigma*5,base_sigma*10]
    dir_path = os.path.dirname(path)
    if("sigma" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"sigma"))
    file_name = os.path.basename(path)

    procs :list[mp.Process] = []
    test_name = "sigma"
    for sigma in sigmas:
        noise_dev_info=f"sigma={sigma}_"
        test_path = os.path.join(dir_path,test_name) 
        hnsw_path = os.path.join(dir_path,test_name,noise_dev_info+file_name.replace(".csv",".hnsw"))
        hnswh_path = os.path.join(dir_path,test_name,noise_dev_info+file_name.replace(".csv",".hnswh"))
        
        if(os.path.basename(hnsw_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnsw_path,sigma,M,efConstr,False))
            procs.append(p)

        if(os.path.basename(hnswh_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnswh_path,sigma,M,efConstr,True))
            procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()

def clean():
    dataset="randvec"   
    for dir_path in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_path)
        
        for f in os.listdir(dir_path):
            if(f.endswith(".hnsw") or f.endswith(".hnswh")):
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)

if __name__ == "__main__":
    exps = (3,4)
    for exp in exps:
        #build_from_file(os.path.join("randvec",f"10^{exp}",f"randvec128_10^{exp}.csv"))
        test_randvec(exp)
    #clean()