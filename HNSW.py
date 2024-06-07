import numpy as np,time,networkx as nx
from queue import PriorityQueue
from HGraph import HGraph,DPUPHGraph,test_run
import random
from tqdm import tqdm
    
class HNSW(HGraph):
    next_seqno:int = 0
    heu_select = False
    def __init__(self, path: str = None,heu_select = False) -> None:
        super().__init__(path)
        self.heu_select = heu_select
    def select_neighbors(self,q:np.ndarray,C:PriorityQueue,M:int,lc:int,extCand:bool = True,keepPrunedConn:bool = True)->PriorityQueue:
        if self.heu_select:
            R = PriorityQueue()#increasing order
            R_size = 0
            W = PriorityQueue()#increasing order
            for c in C.queue:
                W.put(c)
            if extCand:
                for c in C.queue:
                    e = c[1]
                    for e_adj  in self.layers[lc].neighbors(e):
                        if e_adj not in [w[1] for w in W.queue]:
                            d = self.__dist__(q,e_adj)
                            W.put((d,e_adj))

            W_d = PriorityQueue()
            while (not W.empty()) and R_size<M:
                e = W.get()
                if R.empty() or e[0]<R.queue[0][0]:
                    R_size += 1
                    R.put(e)
                else: 
                    W_d.put(e)
            
            if keepPrunedConn:
                while (not W_d.empty()) and R_size<M:
                    R.put(W_d.get())
                    R_size += 1
            
            return R
        else:
            C_ = PriorityQueue()
            i=0
            for c in C.queue:
                if i>=M:
                    break
                C_.put(c)
                i+=1
            return C_

    def insert(self, q:np.array, M:int, M_max:int, efConstr:int, mL:float)->None:

        W = PriorityQueue() #list for the currently found nearest elements
        ep : int = self.ep  #get enter point for hnsw
        L : int = len(self.layers)-1#top layer for hnsw
        l : int= int(-np.log(random.random())*mL)#new element’s level
            
        if(L>l):
            for lc in range(L,l,-1):
                W = self.search_layer(q,ep,1,lc)
                ep = W.queue[0][1]

        for lc in range(min(L,l),-1,-1):
            W = self.search_layer(q,ep,efConstr,lc)
            
            neighbors_q = self.select_neighbors(q,W,M,lc)

            neighbors : list[int] = [n[1] for n in neighbors_q.queue]
            #add bidirectionall connectionts from neighbors to q at layer lc
            self.layers[lc].add_node(self.next_seqno)
            for e in neighbors:
                self.layers[lc].add_edge(self.next_seqno,e)
            #shrink connections if needed
            for e in neighbors:
                eConn : set[int] = set(self.layers[lc].neighbors(e))
                if(len(eConn)>M_max):
                    eConn_q : PriorityQueue = PriorityQueue()
                    for ec in eConn:
                        eConn_q.put((self.__dist__(self.data[e],ec),ec))
                    new_eConn_q = self.select_neighbors(self.data[e],eConn_q,M_max,lc)
                    new_eConn = set(n[1] for n in new_eConn_q.queue)

                    for nec in new_eConn - eConn:
                        self.layers[lc].add_edge(e,nec)
                    
                    for ec in eConn - new_eConn:
                        self.layers[lc].remove_edge(e,ec)
            ep = W.queue[0][1]

        if(l>L):
            while len(self.layers) <l+1:
                L_0 = nx.Graph()
                L_0.add_node(self.next_seqno)
                self.layers.append(L_0)
            self.ep = self.next_seqno
        self.next_seqno+=1
        
    def build(self,M:int,cosine = False,efConstr=100):
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from datafile {self.data_file} ...")
            t = time.time()
            self.cosine = cosine
            self.M = M
            self.M_max = 2*M
            mL:float = 1/(np.log(M))
            for q in tqdm(self.data):
                self.insert(q,self.M,self.M_max,efConstr,mL)
            self.num_of_layers = len(self.layers)
            t = time.time()-t
            
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

class LapUPHNSW(DPUPHGraph, HNSW):       
    def build(self,M:int, cosine = False,efConstr=100):   
        num_unprotected = self.num_of_vectors - self.num_protected
        
        noise = np.random.laplace(0,np.sqrt(self.data.shape[1])/self.epsilon,(self.num_protected,self.data.shape[1]))

        self.data[num_unprotected:] += noise
        HNSW.build(self,M,cosine,efConstr)
        self.data[num_unprotected:] -= noise


if __name__ == '__main__':
    test_run(HNSW)