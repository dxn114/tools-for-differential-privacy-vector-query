import numpy as np,dill,os,time
from graph import Graph
from queue import PriorityQueue
import random,enum
class DistType(enum.Enum):
    EUCLIDIAN = 0
    HAMMING = 1
    COSINE = 2
    
class HNSW:
    data : dict[int,np.array] ={}
    layers : list[Graph] =[]
    num_of_layers : int = 0
    ep : int = 0
    next_seqno:int = 0
    dist_type = DistType.EUCLIDIAN
    algchoose = 3

    def use_heuristic_selection(self)->None:
        self.algchoose = 4
        print("Using heuristic neighbor selection")
    def dist(self,q1,q2):
        if self.dist_type == DistType.EUCLIDIAN:
            return np.sum(np.square(q1 - q2))


    def search_layer(self,q:np.array,ep:int,ef:int,lc:int)->PriorityQueue:
        v = {ep}
        C = PriorityQueue()
        dqep = self.dist(q,self.data[ep])
        C.put((dqep,ep))#increasing order
        W = PriorityQueue()
        W.put((-dqep,ep))#decreasing order
        W_size = 1
        while not C.empty():
            c = C.get()
            f = W.queue[0]
            if c[0] > -f[0]:
                break
            for e in self.layers[lc].neighborhood(c[1]):
                if e not in v:
                    v.add(e)
                    f = W.queue[0]
                    deq = self.dist(self.data[e],q)
                    if deq < -f[0] or W_size<ef:
                        C.put((deq,e))
                        W.put((-deq,e))
                        W_size +=1
                        if W_size>ef:
                            W.get()
                            W_size -=1
        #invert the order of W: make it increasing
        _W = PriorityQueue()
        for w in W.queue:
            _W.put((-w[0],w[1]))
        return _W
    def select_neighbors_simple(self,q:np.array,C:PriorityQueue,M:int)->PriorityQueue:
        C_ = PriorityQueue()
        i=0
        for c in C.queue:
            if i>=M:
                break
            C_.put(c)
            i+=1
        return C_
    def select_neighbors_heuristic(self,q:np.array,C:PriorityQueue,M:int,lc:int,extCand:bool,keepPrunedConn:bool)->PriorityQueue:
        R = PriorityQueue()#increasing order
        R_size = 0
        W = PriorityQueue()#increasing order
        for c in C.queue:
            W.put(c)
        if extCand:
            for c in C.queue:
                e = c[1]
                for e_adj in self.layers[lc].neighborhood(e):
                    if e_adj not in [w[1] for w in W.queue]:
                        d = self.dist(self.data[e_adj],q)
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

    def insert(self, q:np.array, M:int, M_max:int, efConstr:int, mL:float)->None:

        W = PriorityQueue() #list for the currently found nearest elements
        ep : int = self.ep  #get enter point for hnsw
        L : int = self.num_of_layers-1#top layer for hnsw
        l : int= int(-np.log(random.random())*mL)#new element’s level
            
        if(L>l):
            for lc in range(L,l,-1):
                W = self.search_layer(q,ep,1,lc)
                ep = W.queue[0][1]

        for lc in range(min(L,l),-1,-1):
            W = self.search_layer(q,ep,efConstr,lc)
            neighbors_q :PriorityQueue = PriorityQueue()
            
            extConn = True
            keepPrunedConn= True
            algchoose = self.algchoose
            if algchoose==3:
                neighbors_q = self.select_neighbors_simple(q,W,M)
            elif algchoose==4:
                
                neighbors_q = self.select_neighbors_heuristic(q,W,M,lc,extConn,keepPrunedConn)
            neighbors : list[int] = [n[1] for n in neighbors_q.queue]
            #add bidirectionall connectionts from neighbors to q at layer lc
            self.layers[lc].insert_vertex(self.next_seqno)
            for e in neighbors:
                self.layers[lc].insert_edge(self.next_seqno,e)
            #shrink connections if needed
            for e in neighbors:
                eConn : set[int] = self.layers[lc].neighborhood(e)
                if(len(eConn)>M_max):
                    eConn_q : PriorityQueue = PriorityQueue()
                    for ec in eConn:
                        eConn_q.put((self.dist(self.data[e],self.data[ec]),ec))
                    new_eConn_q : PriorityQueue = PriorityQueue()
                    if algchoose==3:
                        new_eConn_q = self.select_neighbors_simple(self.data[e],eConn_q,M_max)
                    elif algchoose==4:
                        new_eConn_q = self.select_neighbors_heuristic(self.data[e],eConn_q,M_max,lc,extConn,keepPrunedConn)
                    new_eConn = [n[1] for n in new_eConn_q.queue]
                    for nec in new_eConn:
                        self.layers[lc].insert_edge(e,nec)
                        if nec in eConn :
                            eConn.discard(nec)
                    
                    for ec in eConn:
                        self.layers[lc].remove_edge(e,ec)
            ep = W.queue[0][1]

        if(l>L):
            while len(self.layers) <l+1:
                L_0 = Graph(self.next_seqno)
                self.layers.append(L_0)
            self.ep = self.next_seqno
        self.num_of_layers = len(self.layers)
        self.next_seqno+=1

    def kNN_search(self,q:np.array,K:int,ef:int)->list[int]:
        print(f"Querying top-{K} for vector data {q} ...")
        t = time.time()
        W = PriorityQueue()
        ep = self.ep
        for lc in range(self.num_of_layers-1,0,-1):
            W : PriorityQueue = self.search_layer(q,ep,1,lc)
            ep = W.queue[0][1]
        W = self.search_layer(q,ep,ef,0)
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
        for seq,datum in self.data.items():
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

    def __init__(self) -> None:
        self.data={}
        self.layers=[]
        
    def build(self,path:str,M:int,efConstr):
        file_name = os.path.basename(path)
        print(f"Building HNSW from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            M_max = 2*M
            mL:float = 1//(np.log(M))
            data = np.loadtxt(path,delimiter=',',dtype=int)
            for vec in data:
                self.data[self.next_seqno]=vec
                self.insert(vec,M,M_max,efConstr,mL)
            t = time.time()-t
            heu = ""
            if self.algchoose == 4:
                heu = "(with heuristic neighbor selection) "
            exp = file_name.removesuffix(".csv")[-1]
            dim = os.path.basename(os.path.dirname(path).removesuffix("d_randvec"))
            print(f"HNSW for 10^{exp} {dim}D vectors {heu}built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")

    def load(self,path:str):
        print(f"Loading HNSW from {os.path.basename(path)} ...")
        if(path.endswith(".hnsw") or path.endswith(".hnswh")):
            with open (path,"rb") as f:
                m : HNSW = dill.load(f)
                self.layers = m.layers
                self.data = m.data
                self.ep  = m.ep
                self.next_seqno = m.next_seqno
                self.num_of_layers = m.num_of_layers
                self.dist_type = m.dist_type
                self.algchoose = m.algchoose

            heu = ""    
            if self.algchoose == 4:
                heu = " (with heuristic neighbor selection)"
            print(f"File {os.path.basename(path)} loaded as HNSW{heu}")
        else:
            print(f"ERROR! Cannot load from file{os.path.basename(path)}")

    def save(self,path) -> None:
        with open(path, "wb") as f:
            dill.dump(self,f)
            print(f"HNSW saved to {os.path.basename(path)}")
