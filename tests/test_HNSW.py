import os,multiprocessing as mp,sys
sys.path.append(os.path.abspath("."))
from randvec import gen_randvec
from HNSW import HNSW

class_name = "HNSW"
ext = f".{class_name.lower()}"
def test_randvec()->None:
    out = open("out.csv", "w")
    dataset=f"randvec_{class_name}"   
    K = 100
    ef= 200
    for i in range(10):
        line = f"Acc{i}/%,"
        exps = [3,4,5]
        dim = 128
        for exp in exps:
            dir_path = os.path.join(dataset,f"10^{exp}")
            for f in os.listdir(dir_path):
                hnsw = HNSW()
                if(f.endswith(ext)):
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
    exp = int(path[-5])
    M = 0
    efConstr = 100
    if exp ==3:
        M = 8
    elif exp ==4:
        M = 16
    elif exp ==5:
        M = 32
    elif exp ==6:
        M = 48

    dir_name = os.path.dirname(path)
    hnsw_path = path.replace(".csv",ext)
    
    if(os.path.basename(hnsw_path) not in os.listdir(dir_name)):
        hnsw1 = HNSW()
        hnsw1.build(path,M,efConstr)
        hnsw1.save(hnsw_path)

def build_for_all():
    dataset=f"randvec_{class_name}"   
    processes : list[mp.Process] = []
    for dir_name in os.listdir(dataset):
        dir_path = os.path.join(dataset,dir_name)
        for f in os.listdir(dir_path):
            if(f.endswith(".csv")):
                file_path = os.path.join(dir_path,f)
                if (f.replace(".csv",ext) not in os.listdir(dir_path)):
                    p = mp.Process(target=build_from_file,args=(file_path,))
                    p.start()
                    processes.append(p)

    for p in processes:
        p.join()
        
dataset_dir=f"randvec_{class_name}" 
def clean():
      
    for dir_name in os.listdir(dataset_dir):
        dir_path = os.path.join(dataset_dir,dir_name)
        
        for f in os.listdir(dir_path):
            if(f.endswith(ext)):
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)


if __name__ == "__main__":
    #build_for_all()
    test_randvec()
    #clean()
    pass
