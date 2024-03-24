import numpy as np,time,os,pickle,networkx as nx,matplotlib.pyplot as plt
from queue import PriorityQueue
from sklearn.metrics import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree

class HGraph:
    data : np.ndarray = np.array([])
    data_file : str = ""
    layers : list[nx.Graph] = []
    num_of_layers : int = 0
    num_of_vectors : int = 0
    ep : int = 0
    M : int = 0
    M_max : int = 0
    file = ""

    def __init__(self,path : str = None) -> None:
        self.layers = []
        if path is not None:
            file_name = os.path.basename(path)
            self.data_file = file_name
            if(path.endswith(".csv") or  path.endswith(".npy")):
                if path.endswith(".csv"):
                    self.data = np.loadtxt(path,delimiter=',')
                elif path.endswith(".npy"):
                    self.data = np.load(path,allow_pickle=True)
                self.num_of_vectors = self.data.shape[0]
            else: 
                print(f"ERROR! Cannot read file{file_name}") 
 
    def __dist__(self,q : np.ndarray,vid:int):
        d = q-self.data[vid]
        return np.linalg.norm(d)

    def search_layer(self,q:np.ndarray,ep:int,ef:int,lc:int)->PriorityQueue:
        v = {ep}
        C = PriorityQueue()
        dqep = self.__dist__(q,ep)
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
                    deq = self.__dist__(q,e)
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
        print(f"Querying top-{K} from {self.file} ...")
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
        return res

    
    def build_layer(self, lc):
        pass     

    def build(self,M:int):
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            l = (-np.log(np.random.rand(self.num_of_vectors))*mL).astype(int)# new element’s level (count from 0)
            self.M = M
            self.M_max = 2*M            
            for i in range(self.num_of_vectors):
                while len(self.layers)-1 < l[i]:
                    self.layers.append(nx.Graph())
                for j in range(l[i]+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            self.ep = int(list(self.layers[-1].nodes())[-1])
            
            for lc in range(self.num_of_layers-1,-1,-1):
                self.build_layer(lc)

            t = time.time()-t
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

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
        self.file = os.path.basename(path)
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

class DPHGraph(HGraph):
    epsilon : float = 0
    def __init__(self,epsilon=1,path : str = None) -> None:
        super().__init__(path)
        self.epsilon = epsilon