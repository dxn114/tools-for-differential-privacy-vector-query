import numpy as np,time,os,pickle,networkx as nx,matplotlib.pyplot as plt
from queue import PriorityQueue
import psutil
from sklearn.metrics import pairwise_distances

class HGraph:
    data : np.ndarray = np.zeros(0)
    layers : list[nx.Graph] = []
    num_of_layers : int = 0
    num_of_vectors : int = 0
    ep : int = 0
    M_max : int = 0
    Dist_Mat : np.ndarray = np.zeros(0)

    def __init__(self) -> None:
        self.layers = []
    def dist_square_sum(self,x : np.ndarray, lc : int, A : nx.Graph) -> float:
        # x gives the order of nodes
        vector_ids = [self.layers[lc].nodes()[i] for i in x]
        H : float = 0
        for u,v in A.edges:
            v1 = vector_ids[u]
            v2 = vector_ids[v]
            H += self.dist(self.data[v1],v2,v1)
        return H
    
    def precal_dist(self)->None:
        size = self.data.shape[0]
        estimated_mem = (size**2)*4
        if estimated_mem > psutil.virtual_memory().available:
            return
        self.Dist_Mat = pairwise_distances(self.data,metric='euclidean',n_jobs=-1)
    def dist(self,q : np.ndarray,vid:int,qid:int=None):
        if qid == None or self.Dist_Mat.size == 0:
            d = q-self.data[vid]
            return np.linalg.norm(d)
        else:
            return self.Dist_Mat[vid,qid]

    def search_layer(self,q:np.ndarray,ep:int,ef:int,lc:int,qid : int = None)->PriorityQueue:
        v = {ep}
        C = PriorityQueue()
        dqep = self.dist(q,ep,qid)
        C.put((dqep,ep))#increasing order
        W = PriorityQueue()
        W.put((-dqep,ep))#decreasing order
        W_size = 1
        while not C.empty():
            c = C.get()
            f = W.queue[0]
            if c[0] > -f[0]:
                break
            for e in self.layers[lc].neighbors(c[1]):
                if e not in v:
                    v.add(e)
                    f = W.queue[0]
                    deq = self.dist(q,e,qid)
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
    
    def kNN_search(self,q:np.ndarray,K:int,ef:int)->list[int]:
        print(f"Querying top-{K} ...")
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
    
    def real_kNN(self,q:np.ndarray,K:int)->list:
        t = time.time()
        dist_vec = pairwise_distances(np.array([q]),self.data,metric='euclidean',n_jobs=-1).ravel()
        res = np.argsort(dist_vec)[:K].tolist()
        t = time.time()-t
        print(f"Real result retrieved in {t:.3f} seconds.")
        return res

    
    def build_layer(self, lc):
        pass

    def build(self,path:str,M:int):
        file_name = os.path.basename(path)
        class_name = self.__class__.__name__
        print(f"Building {class_name} from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            self.M_max = 2*M
            mL:float = 1/(np.log(M))
            self.data = np.loadtxt(path,delimiter=',',dtype=int)
            
            # self.precal_dist()
            
            self.num_of_vectors = self.data.shape[0]

            for i in range(self.num_of_vectors):
                L = len(self.layers)-1
                l : int= int(-np.log(np.random.rand(1)[0])*mL)#new element’s level
                if L<l:
                    self.ep = i

                while len(self.layers)-1<l:
                    self.layers.append(nx.Graph())

                for j in range(l+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            #Procs = []
            for lc in range(self.num_of_layers-1,-1,-1):
                #proc = mp.Process(target=self.build_layer,args=(lc,))
                #Procs.append(proc)
                #proc.start()
                self.build_layer(lc)

            #for proc in Procs:
            #    proc.join()
            self.Dist_Mat = np.zeros(0)
            dim = self.data.shape[1]
            t = time.time()-t
            print(f"{class_name} for {self.num_of_vectors} {dim}D vectors built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")

    def load(self,path:str):
        class_name = self.__class__.__name__
        ext = '.'+class_name.lower()
        print(f"Loading {class_name} from {os.path.basename(path)} ...")
        if(path.endswith(ext)):
            with open (path,"rb") as f:
                m : HGraph = pickle.load(f)
                self.__dict__.update(m.__dict__)
 
            print(f"File {os.path.basename(path)} loaded as {class_name}.")
        else:
            print(f"ERROR! Cannot load from file{os.path.basename(path)}")

    def save(self,path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self,f)
            print(f"{self.__class__.__name__} saved to {os.path.basename(path)}")

    def draw(self,path):
        if "layer_view" not in os.listdir(path):
            os.mkdir(os.path.join(path,"layer_view"))
        else:
            for jpg in os.listdir(os.path.join(path,"layer_view")):
                if  jpg.endswith(".jpg"):
                    os.remove(os.path.join(path,"layer_view",jpg))
        from math import sqrt
        for i in range(self.num_of_layers):
            n = self.layers[i].number_of_nodes()
            l = int(sqrt(n))*10
            plt.figure(figsize=(l, l))
            nx.draw_networkx(self.layers[i],pos=nx.spring_layout(self.layers[i],k=n**2,scale=n**4))
            plt.savefig(os.path.join(path,"layer_view",f"layer{i}.jpg"))
            plt.clf()
