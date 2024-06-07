import numpy as np,time,os,pickle,networkx as nx,matplotlib.pyplot as plt
from queue import PriorityQueue
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    cosine = False

    def __init__(self,path : str = None) -> None:
        self.layers = []
        if path is not None:
            file_name = os.path.basename(path)
            self.data_file = file_name
            if path.endswith(".npy"):
                self.data = np.load(path,allow_pickle=True)
                self.num_of_vectors = self.data.shape[0]
            else: 
                print(f"ERROR! Cannot initialize from file {file_name}") 
 
    def __dist__(self,q : np.ndarray,vid:int):
        # return distance between query q and vector vid in the model
        if self.cosine:
            return cosine(q,self.data[vid])
        else:
            return euclidean(q,self.data[vid])

    def search_layer(self,q:np.ndarray,ep:int,ef:int,lc:int)->PriorityQueue:
        v = {ep}
        C = PriorityQueue()
        dqep = self.__dist__(q,ep)
        C.put((dqep,ep))#increasing order
        W = PriorityQueue()
        W.put((-dqep,ep))#decreasing order
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
                    if deq < -f[0] or len(W.queue)<ef:
                        C.put((deq,e))
                        W.put((-deq,e))

                        if len(W.queue)>ef:
                            W.get()
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
        vids = []
        dist_k = []
        for _ in range(K):
            if W.empty():
                break
            dist,vid = W.get()
            vids.append(vid)
            dist_k.append(dist)
        t = time.time()-t
        print(f"Search result retrieved in {t:.3f} seconds.\nCalculating accuracy ...")
        return vids, dist_k
    
    def real_kNN(self,q:np.ndarray,K:int)->list:
        t = time.time()
        dist_vec = pairwise_distances(q.reshape(1,-1),self.data,n_jobs=-1,metric="cosine" if self.cosine else "euclidean")[0]
        vids = np.argpartition(dist_vec,K)[:K]
        dist_k = dist_vec[vids]
        order = np.argsort(dist_k)
        vids = vids[order]
        dist_k = dist_k[order]
        t = time.time()-t
        return vids.tolist(), dist_k.tolist()

    
    def build_layer(self, lc):
        pass     

    def build(self,M:int,cosine = False):
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            l = (-np.log(np.random.rand(self.num_of_vectors))*mL).astype(int)# new element’s level (count from 0)
            self.M = M
            self.M_max = 2*M        
            self.cosine = cosine   
            
            self.layers = [nx.Graph() for _ in range(l.max()+1)]    

            for i in range(self.num_of_vectors):
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

class DPUPHGraph(HGraph):
    epsilon : float = 0
    def __init__(self, epsilon=1, path:str = None, path2:str = None, num_unprotected = None) -> None:
        self.epsilon = epsilon
        if path == None:
            self.layers = []
        else:
            if path2 == None:
                super().__init__(path=path)            
                if num_unprotected == None or num_unprotected < 0: 
                    num_unprotected = int(self.num_of_vectors/2)
                elif num_unprotected < 1:
                    num_unprotected = int(self.num_of_vectors*num_unprotected)
                self.num_protected = self.num_of_vectors - num_unprotected
            else:
                self.layers = []
                file1 = os.path.basename(path)
                file2 = os.path.basename(path2)
                self.data_file = file1 + " and " + file2
                if path.endswith(".npy") and path2.endswith(".npy"):
                    data1 = np.load(path,allow_pickle=True)
                    data2 = np.load(path2,allow_pickle=True)
                    self.data = np.stack((data1,data2),axis=0)
                    self.num_of_vectors = self.data.shape[0]
                    self.num_protected = data2.shape[0]
                else: 
                    print(f"ERROR! Cannot initialize from file {self.data_file}") 


def test_run(test_class, dataset="randvec",exp=3):
    class_name = test_class.__name__
    dir_path = os.path.join(dataset,f"10^{exp}") 
    npy_path = os.path.join(dir_path,f"{dataset}_10^{exp}.npy") 
    h = test_class(path=npy_path)
    h.build(16)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = test_class()
    n.load(h_path)
    # n.draw(dir_path)
    return n

def DPknn(queries : torch.Tensor,data : torch.Tensor,epsilon:float,k:int,cosine : bool = False,noise : str = "gumbel")->np.ndarray: # Assume q and data are disjoint
    batch_size = 1024
    queries = TensorDataset(queries)
    queries = DataLoader(queries,batch_size=batch_size)
    k_smallest_indices = []
    for q in queries:
        q = q[0]
        if cosine:
            dist = 1 - torch.nn.functional.cosine_similarity(q,data,dim=1)
        else:
            dist = torch.cdist(q,data)
        
        dist = dist.cpu().numpy()
        if noise == "gumbel":
            dist -= np.random.gumbel(0,2*k/epsilon,size=dist.shape)
        elif noise == "laplace":
            dist -= np.random.laplace(0,2*k/epsilon,size=dist.shape)
        elif noise == "exponential":
            dist -= np.random.exponential(2*k/epsilon,size=dist.shape)
        else:
            raise ValueError("Unsupported noise model")
        dist = torch.tensor(dist,device=device)
        k_smallest_indices.append(torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy())
    return np.concatenate(k_smallest_indices)