import sys,os,multiprocessing as mp,matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from randvec import gen_randvec
from DPHREG import DPHREG
class_name = "DPHREG"
ext = f".{class_name.lower()}"
def test_randvec(exp)->None:
    out = open(f"out({class_name}).csv", "w")
    dataset=os.path.join(f"randvec_{class_name}",f"10^{exp}")   
    K = 100
    ef= 200
    test_sets = ["epsilon","delta"]
    line = "file_name,"
    for i in range(10):
        line += f"Recall{i},"
    line +="Avg\n"
    out.write(line)
    for test_set in test_sets:
        dir_path = os.path.join(dataset,test_set)
        avgs = {}
        for f in os.listdir(dir_path):
            h = DPHREG()
            if f.endswith(ext):
                sum = 0
                line = f+','
                model_path = os.path.join(dir_path,f)
                h.load(model_path)

                for t in range(10):
                    q = gen_randvec(128)
                    print("======================================")
            
                    res = h.kNN_search(q,K,ef)
                    real = h.real_kNN(q,K)
                    
                    score = len(set(res)&set(real))
                    
                    rec = score/float(K)
                    sum += rec
                    print(f"Recall: {rec}")
                    line += f"{rec},"

                avg = sum/10.0
                if test_set=="epsilon":
                    avgs[h.epsilon] = avg
                elif test_set=="delta":
                    avgs[h.delta] = avg
                line += f"{avg}\n"
                out.write(line)
        
        avgs = dict(sorted(avgs.items()))

        plt.ylabel("Recall")
        plt.xlabel(test_set)
        plt.ylim(0,1.1)
        plt.plot(avgs.keys(),avgs.values())
        plt.savefig(os.path.join(dataset,f"{test_set}.png"))
        plt.close()

    out.close()
        

def build_model(vecfile_path,model_path,epsilon=0.2,delta=0.05):
    exp = int(vecfile_path[-5])
    M = 0
    if exp ==3:
        M = 8
    elif exp ==4:
        M = 32
    elif exp ==5:
        M = 64
    elif exp ==6:
        M = 128
    
    h = DPHREG(epsilon=epsilon,delta=delta)
    h.build(vecfile_path,M)
    h.save(model_path)

def build_from_file(vecfile_path:str):
    epsilons = [0.1,0.2,0.5,1]
    deltas = [0.01,0.02,0.05,0.1]
    dir_path = os.path.dirname(vecfile_path)
    if("epsilon" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"epsilon"))
    if("delta" not in os.listdir(dir_path)):
        os.mkdir(os.path.join(dir_path,"delta"))
    file_name = os.path.basename(vecfile_path)

    procs = []
    test_name = "epsilon"
    for epsilon in epsilons:
        priv_budget_info=f"epsilon={epsilon}_"
        test_path = os.path.join(dir_path,test_name) 
        h_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",ext))
        
        if(os.path.basename(h_path) not in os.listdir(test_path)):

            p=mp.Process(target=build_model,args=(vecfile_path,h_path,epsilon,0.05))
            procs.append(p)


    test_name = "delta"
    for delta in deltas:
        priv_budget_info=f"delta={delta}_"
        test_path = os.path.join(dir_path,test_name) 
        h_path = os.path.join(dir_path,test_name,priv_budget_info+file_name.replace(".csv",ext))
        
        if(os.path.basename(h_path) not in os.listdir(test_path)):
            p=mp.Process(target=build_model,args=(vecfile_path,h_path,0.2,delta))
            procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()

datasets_dir=f"randvec_{class_name}"  

if __name__ == "__main__":
    exps = [3,4,5]
    for exp in exps:
        build_from_file(os.path.join(datasets_dir,f"10^{exp}",f"randvec128_10^{exp}.csv"))
        test_randvec(exp)
    
    pass