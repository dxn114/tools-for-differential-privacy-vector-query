import numpy as np,os
from queue import PriorityQueue
from HNSW import HNSW
import random
    
class HNSWH(HNSW):
    def select_neighbors(self,q:np.ndarray,C:PriorityQueue,M:int,lc:int,extCand:bool,keepPrunedConn:bool)->PriorityQueue:
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
        
if __name__ == '__main__':
    class_name = HNSWH.__name__
    dir_path = os.path.join(f"randvec","10^3") 
    npy_path = os.path.join(dir_path,"randvec_10^3.npy")
    h = HNSWH(npy_path)
    h.build(8,100)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = HNSWH()
    n.load(h_path)
    #n.draw(dir_path)
    pass