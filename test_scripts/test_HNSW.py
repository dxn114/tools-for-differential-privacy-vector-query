from randvec import gen_randvec
from HNSW import HNSW
import os,multiprocessing as mp
    
def test_randvec()->None:
    out = open("out.csv", "w")
    dataset="randvec"   
    K = 50
    ef= 200
    for i in range(10):
        line = f"Acc{i}/%,"
        dims = [10,100,200]
        for dim in dims:
            dir_path = os.path.join(dataset,f"{dim}d_randvec")
            for f in os.listdir(dir_path):
                hnsw = HNSW()
                if(f.endswith(".hnsw") or f.endswith(".hnswh")):
                    q = gen_randvec(dim)
                    print("======================================")
                    model_path = os.path.join(dir_path,f)
                    hnsw.load(model_path)
                    res = hnsw.kNN_search(q,K,ef)
                    real = hnsw.real_kNN(q,K)
                    
                    score = 0
                    for i in range(K):
                        if res[i] in real:
                            score +=1
                    print(f"Accuracy: {score*100//K}%")
                    line += f"{score*100//K},"

        line = line.removesuffix(",") + '\n'
        out.write(line)
        

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

    dir_name = os.path.dirname(path)
    hnsw_path = path.replace(".csv",".hnsw")
    hnswh_path = path.replace(".csv",".hnswh")
    
    if(os.path.basename(hnsw_path) not in os.listdir(dir_name)):
        hnsw1 = HNSW()
        hnsw1.build(path,M,efConstr)
        hnsw1.save(hnsw_path)

    if(os.path.basename(hnswh_path) not in os.listdir(dir_name)):
        hnsw2 = HNSW()
        hnsw2.use_heuristic_selection()  
        hnsw2.build(path,M,efConstr)
        hnsw2.save(hnswh_path)

def build_for_all():
    dataset="randvec"   
    processes : list[mp.Process] = []
    for dir_name in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_name)
        for f in os.listdir(dir_path):
            if(f.endswith(".csv")):
                file_path = os.path.join(dir_path,f)
                if (f.replace(".csv",".hnsw") not in os.listdir(dir_path)) or (f.replace(".csv",".hnswh") not in os.listdir(dir_path)):
                    p = mp.Process(target=build_from_file,args=(file_path,))
                    p.start()
                    processes.append(p)

    for p in processes:
        p.join()

def clean():
    dataset="randvec"   
    for dir_name in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_name)
        
        for f in os.listdir(dir_path):
            if(f.endswith(".hnsw") or f.endswith(".hnswh")):
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)


if __name__ == "__main__":
    build_for_all()
    #test_randvec()
    #clean()
    pass
