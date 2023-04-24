from HNSW import HNSW
import numpy as np
import os,time,dill
from queue import PriorityQueue

def l2_sensitivity(P:np.ndarray):
    res = 0
    for row in P:
        norm = np.linalg.norm(row)
        if norm > res:
            res = norm
    return res

class JLHNSW (HNSW):
    raw_data : dict[int,np.array] ={}
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

    def __init__(self) -> None:
        self.data={}
        self.layers=[]
        self.raw_data = {}
        
    def set_priv_budget(self,epsilon,delta):
        self.epsilon = epsilon
        self.delta = delta


    def build(self,path:str,M:int,efConstr):
        file_name = os.path.basename(path)
        print(f"Building HNSW from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            M_max = 2*M
            mL:float = 1//(np.log(M))
            raw_data = np.loadtxt(path,delimiter=',',dtype=int)
            dim = raw_data.shape[1]
            #projected dimension
            k_OPT = self.k_OPT
            #projection matrix
            self.Proj_Mat = np.random.normal(0,1.0/np.sqrt(k_OPT),(dim,k_OPT))
            w2Proj_Max = l2_sensitivity(self.Proj_Mat)
            self.sigma = w2Proj_Max*(np.sqrt(2*(self.epsilon - np.log(2*self.delta)))//self.epsilon)
            #noise matrix
            Noise_Mat = np.random.normal(0,self.sigma,(raw_data.shape[0],k_OPT))
            data:np.ndarray = np.add(np.matmul(raw_data,self.Proj_Mat),Noise_Mat)
            
            for i in range(data.shape[0]):
                self.raw_data[self.next_seqno]=raw_data[i]
                self.data[self.next_seqno]=data[i]
                self.insert(data[i],M,M_max,efConstr,mL)
            t = time.time()-t
            heu = ""
            if self.algchoose == 4:
                heu = "(with heuristic neighbor selection) "
            exp = file_name.removesuffix(".csv")[-1]
            print(f"HNSW for 10^{exp} {dim}D vectors {heu}built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")

    def kNN_search(self,q:np.array,K:int,ef:int)->list[int]:
        print(f"Querying top-{K} for vector data {q} ...")
        q_ = np.matmul(q,self.Proj_Mat)
        q_ = np.add(q_,np.random.normal(0,self.sigma,(1,self.Proj_Mat.shape[1])))
        t = time.time()
        W = PriorityQueue()
        ep = self.ep
        for lc in range(self.num_of_layers-1,0,-1):
            W : PriorityQueue = self.search_layer(q_,ep,1,lc)
            ep = W.queue[0][1]
        W = self.search_layer(q_,ep,ef,0)
        W_ = []
        for i in range(K):
            W_.append(W.get()[1])
        t = time.time()-t
        print(f"Search result retrieved in {t:.3f} seconds.\nCalculating accuracy ...")
        return W_
    
    def real_kNN(self,q:np.array,K:int)->list:
        t = time.time()
        Q = PriorityQueue()
        i=0
        for seq,datum in self.raw_data.items():
            d = self.dist(q,datum)
            if(i<K):
                Q.put((-d,seq))
                i+=1
            elif d<-Q.queue[0][0]:
                Q.get()
                Q.put((-d,seq))
        res = []
        for i in range(K):
            res.insert(0,Q.get()[1])
        t = time.time()-t
        print(f"Real result retrieved in {t:.3f} seconds.")
        return res
    
    def load(self,path:str):
        print(f"Loading HNSW from {os.path.basename(path)} ...")
        if(path.endswith(".hnsw") or path.endswith(".hnswh")):
            with open (path,"rb") as f:
                m : JLHNSW = dill.load(f)
                self.layers = m.layers
                self.data = m.data
                self.ep  = m.ep
                self.next_seqno = m.next_seqno
                self.num_of_layers = m.num_of_layers
                self.dist_type = m.dist_type
                self.algchoose = m.algchoose
                self.raw_data = m.raw_data
                self.Proj_Mat = m.Proj_Mat
                self.epsilon = m.epsilon
                self.delta = m.delta
                self.sigma = m.sigma
                self.k_OPT = m.k_OPT
                
            heu = ""    
            if self.algchoose == 4:
                heu = " (with heuristic neighbor selection)"
            print(f"File {os.path.basename(path)} loaded as HNSW{heu}")
        else:
            print(f"ERROR! Cannot load from file{os.path.basename(path)}")