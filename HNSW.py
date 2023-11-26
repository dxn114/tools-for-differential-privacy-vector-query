import numpy as np,os,time,networkx as nx
from queue import PriorityQueue
from HGraph import HGraph
import random
from tqdm import trange
    
class HNSW(HGraph):
    next_seqno:int = 0
    def select_neighbors(self,q:np.ndarray,C:PriorityQueue,M:int,lc:int,extCand:bool,keepPrunedConn:bool,qid:int = None)->PriorityQueue:
        C_ = PriorityQueue()
        i=0
        for c in C.queue:
            if i>=M:
                break
            C_.put(c)
            i+=1
        return C_

    def insert(self, q:np.array, M:int, M_max:int, efConstr:int, mL:float,qid:int = None)->None:

        W = PriorityQueue() #list for the currently found nearest elements
        ep : int = self.ep  #get enter point for hnsw
        L : int = len(self.layers)-1#top layer for hnsw
        l : int= int(-np.log(random.random())*mL)#new element’s level
            
        if(L>l):
            for lc in range(L,l,-1):
                W = self.search_layer(q,ep,1,lc,qid)
                ep = W.queue[0][1]

        for lc in range(min(L,l),-1,-1):
            W = self.search_layer(q,ep,efConstr,lc,qid)
            neighbors_q :PriorityQueue = PriorityQueue()

            extConn = True
            keepPrunedConn= True
            
            neighbors_q = self.select_neighbors(q,W,M,lc,extConn,keepPrunedConn,qid)

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
                        eConn_q.put((self.dist(self.data[e],ec,qid=e),ec))
                    new_eConn_q : PriorityQueue = PriorityQueue()
                    new_eConn_q = self.select_neighbors(self.data[e],eConn_q,M_max,lc,extConn,keepPrunedConn,qid=e)
                    new_eConn = [n[1] for n in new_eConn_q.queue]
                    for nec in new_eConn:
                        self.layers[lc].add_edge(e,nec)
                        if nec in eConn :
                            eConn.discard(nec)
                    
                    for ec in eConn:
                        self.layers[lc].remove_edge(e,ec)
            ep = W.queue[0][1]

        if(l>L):
            while len(self.layers) <l+1:
                L_0 = nx.Graph()
                L_0.add_node(self.next_seqno)
                self.layers.append(L_0)
            self.ep = self.next_seqno
        self.next_seqno+=1
        
    def build(self,path:str,M:int,efConstr):
        file_name = os.path.basename(path)
        class_name = self.__class__.__name__
        print(f"Building {class_name} from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            self.M_max = 2*M
            mL:float = 1/(np.log(M))
            self.data = np.loadtxt(path,delimiter=',',dtype=int)
            self.num_of_vectors = self.data.shape[0]
            self.precal_dist()
            for vid in trange(self.num_of_vectors):
                self.insert(self.data[vid],M,self.M_max,efConstr,mL,qid=vid)
            self.num_of_layers = len(self.layers)
            t = time.time()-t
            dim = self.data.shape[1]
            
            print(f"{class_name} for {self.num_of_vectors} {dim}D vectors built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")

if __name__ == '__main__':
    h = HNSW()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16,100)
    h.save(h_path)
    n = HNSW()
    n.load(h_path)
    #n.draw(dir_path)
    pass