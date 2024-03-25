import numpy as np,os
from queue import PriorityQueue
from HNSW import HNSW
from HGraph import test_run
    
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
    test_run(HNSWH)