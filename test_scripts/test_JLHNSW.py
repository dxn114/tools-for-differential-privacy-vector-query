import sys,os,multiprocessing as mp,matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from randvec import gen_randvec
from JLHNSW import JLHNSW
    
def test_randvec(exp)->None:
    out = open("out(JL).csv", "w")
    dataset=os.path.join("randvec_JLHNSW",f"10^{exp}")   
    K = 100
    ef= 200
    test_sets = ["epsilon","delta","k_OPT"]
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
            hnsw = JLHNSW()
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
                if test_set=="epsilon":
                    if hnsw.algchoose==3:
                        avgs[hnsw.epsilon] = avg
                    if hnsw.algchoose==4:
                        avgs_h[hnsw.epsilon] = avg
                elif test_set=="delta":
                    if hnsw.algchoose==3:
                        avgs[hnsw.delta] = avg
                    if hnsw.algchoose==4:
                        avgs_h[hnsw.delta] = avg
                elif test_set=="k_OPT":
                    if hnsw.algchoose==3:
                        avgs[hnsw.k_OPT] = avg
                    if hnsw.algchoose==4:
                        avgs_h[hnsw.k_OPT] = avg
                line += f"{avg}\n"
                out.write(line)
        
        avgs = dict(sorted(avgs.items()))
        avgs_h = dict(sorted(avgs_h.items()))

        plt.ylabel("Accuracy/%")
        plt.xlabel(test_set)
        plt.plot(avgs.keys(),avgs.values(),label = "hnsw")
        plt.plot(avgs_h.keys(),avgs_h.values(), label = "hnswh")
        plt.legend()
        plt.savefig(os.path.join(dataset,f"{test_set}.png"))
        plt.close()

    out.close()
        

def build_JLHNSW(vecfile_path,model_path,k_OPT,epsilon,delta,M,efConstr,use_heu):
    hnsw = JLHNSW()
    hnsw.set_priv_budget(epsilon,delta)
    hnsw.set_k_OPT(k_OPT)
    if(use_heu==True):
        hnsw.use_heuristic_selection()  
    hnsw.build(vecfile_path,M,efConstr)
    hnsw.save(model_path)

def build_from_file(path:str):
    scale = int(path[-5])
    M = 0
    efConstr = 100
    if scale ==3:
        M = 8
    elif scale ==4:
        M = 16
    elif scale ==5:
        M = 32
    elif scale ==6:
        M = 48

    epsilons = [1,2,5,10]
    deltas = [0.01,0.02,0.05,0.1]
    k_OPTs = [36,49,64,81,100]
    dir_path = os.path.dirname(path)
    if("epsilon" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"epsilon"))
    if("delta" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"delta"))
    if("k_OPT" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"k_OPT"))
    file_name = os.path.basename(path)

    procs = []
    procs_h = []
    test_name = "epsilon"
    for epsilon in epsilons:
        delta = 0.01
        k_OPT = 96
        priv_budget_info=f"epsilon={epsilon}_"
        test_path = os.path.join(dir_path,test_name) 
        hnsw_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnsw"))
        hnswh_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnswh"))
        
        if(os.path.basename(hnsw_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnsw_path,k_OPT,epsilon,delta,M,efConstr,False))
            procs.append(p)

        if(os.path.basename(hnswh_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnswh_path,k_OPT,epsilon,delta,M,efConstr,True))
            procs_h.append(p)

    test_name = "delta"
    for delta in deltas:
        epsilon=1
        k_OPT = 96
        priv_budget_info=f"delta={delta}_"
        test_path = os.path.join(dir_path,test_name) 
        hnsw_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnsw"))
        hnswh_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnswh"))
        
        if(os.path.basename(hnsw_path) not in os.listdir(test_path)):
            p=mp.Process(target=build_JLHNSW,args=(path,hnsw_path,k_OPT,epsilon,delta,M,efConstr,False))
            procs.append(p)

        if(os.path.basename(hnswh_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnswh_path,k_OPT,epsilon,delta,M,efConstr,True))
            procs_h.append(p)

    test_name = "k_OPT"
    for k_OPT in k_OPTs:
        epsilon = 1
        delta = 0.01
        priv_budget_info=f"k_OPT={k_OPT}_"
        test_path = os.path.join(dir_path,test_name) 
        hnsw_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnsw"))
        hnswh_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",".hnswh"))
        
        if(os.path.basename(hnsw_path) not in os.listdir(test_path)):
            p=mp.Process(target=build_JLHNSW,args=(path,hnsw_path,k_OPT,epsilon,delta,M,efConstr,False))
            procs.append(p)

        if(os.path.basename(hnswh_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_JLHNSW,args=(path,hnswh_path,k_OPT,epsilon,delta,M,efConstr,True))
            procs_h.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    #for p in procs_h:
    #    p.start()
    #for p in procs_h:
    #    p.join()
datasets_dir="randvec_JLHNSW"  
def clean():    
    for dir_path in os.listdir(datasets_dir):
        dir_path = os.path.join(datasets_dir,dir_path)
        
        for f in os.listdir(dir_path):
            if(f.endswith(".hnsw") or f.endswith(".hnswh")):
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)

if __name__ == "__main__":
    exps = [4,5]
    for exp in exps:
        build_from_file(os.path.join(datasets_dir,f"10^{exp}",f"randvec128_10^{exp}.csv"))
        test_randvec(exp)
    #clean()
    pass