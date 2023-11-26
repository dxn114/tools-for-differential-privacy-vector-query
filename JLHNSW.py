from HNSW import HNSW
import numpy as np
import os,time
from queue import PriorityQueue
from tqdm import trange
import psutil
from sklearn.metrics import pairwise_distances

def l2_sensitivity(P:np.ndarray):
    res = 0
    for row in P:
        norm = np.linalg.norm(row)
        if norm > res:
            res = norm
    return res

class JLHNSW (HNSW):
    noisy_data : np.ndarray = np.zeros(0)
    Proj_Mat : np.ndarray = np.zeros((0))
    #privacy budget
    epsilon = 1
    delta = 0.01
    sigma = -1
    k_OPT = 100

    def set_k_OPT(self, k_OPT):
        self.k_OPT=k_OPT
    def set_priv_budget(self,epsilon,delta):
        self.epsilon = epsilon
        self.delta = delta

    def dist(self,q : np.ndarray,vid:int,qid:int=None):
        if qid == None:
            d = q-self.noisy_data[vid]
            return np.inner(d,d)
        else:
            return self.Dist_Mat[vid,qid]
    def precal_dist(self)->None:
        size = self.noisy_data.shape[0]
        estimated_mem = (size**2)*4
        if estimated_mem > psutil.virtual_memory().available:
            return
        self.Dist_Mat = pairwise_distances(self.noisy_data,metric='euclidean',n_jobs=-1)
    def build(self,path:str,M:int,efConstr):
        file_name = os.path.basename(path)
        class_name = self.__class__.__name__
        print(f"Building {class_name} from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            M_max = 2*M
            mL:float = 1//(np.log(M))
            self.data = np.loadtxt(path,delimiter=',',dtype=int)
            dim = self.data.shape[1]
            #projected dimension
            k_OPT = self.k_OPT
            #projection matrix
            self.Proj_Mat = np.random.normal(0,1.0/np.sqrt(k_OPT),(dim,k_OPT))
            w2Proj_Max = l2_sensitivity(self.Proj_Mat)
            self.sigma = w2Proj_Max*(np.sqrt(2*(self.epsilon - np.log(2*self.delta)))//self.epsilon)
            #noise matrix
            Noise_Mat = np.random.normal(0,self.sigma,(self.data.shape[0],k_OPT))
            self.noisy_data:np.ndarray = np.add(np.matmul(self.data,self.Proj_Mat),Noise_Mat)
            self.num_of_vectors = self.noisy_data.shape[0]

            self.precal_dist()

            for vid in trange(self.num_of_vectors):
                self.insert(self.noisy_data[vid],M,M_max,efConstr,mL,qid=vid)

            self.num_of_layers = len(self.layers)
            t = time.time()-t

            exp = file_name.removesuffix(".csv")[-1]
            print(f"{class_name} for 10^{exp} {dim}D vectors built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")

    def kNN_search(self,q:np.array,K:int,ef:int)->list[int]:
        q_ = np.matmul(q,self.Proj_Mat)
        q_ = np.add(q_,np.random.normal(0,self.sigma,(1,self.Proj_Mat.shape[1])))
        
        return super().kNN_search(q_,K,ef)

if __name__ == '__main__':
    h = JLHNSW()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16,100)
    h.save(h_path)
    n = JLHNSW()
    n.load(h_path)
    n.draw(dir_path)
    pass