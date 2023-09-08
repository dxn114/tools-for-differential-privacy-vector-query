import numpy as np,time,os,dill,multiprocessing as mp,networkx as nx,matplotlib.pyplot as plt
from queue import PriorityQueue

def dist(q1,q2):
    d = q1-q2
    return np.inner(d,d)
    
         
class HGraph:
    data : np.ndarray = np.zeros(0)
    layers : list[nx.Graph] = []
    num_of_layers : int = 0
    num_of_vectors : int = 0
    ep : int = 0
    M_max : int = 0
    def search_layer(self,q:np.array,ep:int,ef:int,lc:int)->PriorityQueue:
        v = {ep}
        C = PriorityQueue()
        dqep = dist(q,self.data[ep])
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
                    deq = dist(self.data[e],q)
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
        for vector_id in range(self.data.shape[0]):
            d = dist(q,self.data[vector_id])
            if(i<K):
                Q.put((-d,vector_id))
                i+=1
            elif d<-Q.queue[0][0]:
                Q.get()
                Q.put((-d,vector_id))
        res = []
        for i in range(K):
            res.insert(0,Q.get()[1])
        t = time.time()-t
        print(f"Real result retrieved in {t:.3f} seconds.")
        return res

    
    def build_layer(self, lc):
        pass

    def build(self,path:str,M:int):
        file_name = os.path.basename(path)
        print(f"Building {self.__class__.__name__} from datafile {file_name} ...")
        if(path.endswith(".csv")):
            t = time.time()
            self.M_max = 2*M
            mL:float = 1/(np.log(M))
            self.data = np.loadtxt(path,delimiter=',',dtype=int)
            
            t = time.time()-t
            self.num_of_vectors = self.data.shape[0]

            for i in range(self.num_of_vectors):
                l : int= int(-np.log(np.random.rand(1)[0])*mL)#new element’s level
                if len(self.layers)<l+1:
                    self.ep = i

                while len(self.layers)<l+1:
                    self.layers.append(nx.Graph())

                for j in range(l+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            #Procs = []
            for lc in range(self.num_of_layers):
                #proc = mp.Process(target=self.build_layer,args=(lc,layer_vector_ids[lc]))
                #Procs.append(proc)
                #proc.start()
                self.build_layer(lc)

            #for proc in Procs:
            #    proc.join()
            
            dim = self.data.shape[1]
            print(f"{self.__class__.__name__} for {self.num_of_vectors} {dim}D vectors built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! Cannot build from file{file_name}")    

    def load(self,path:str):
        class_name = self.__class__.__name__
        ext = '.'+class_name.lower()
        print(f"Loading {class_name} from {os.path.basename(path)} ...")
        if(path.endswith(ext)):
            with open (path,"rb") as f:
                m : HGraph = dill.load(f)
                self.__dict__.update(m.__dict__)
 
            print(f"File {os.path.basename(path)} loaded as {class_name}.")
        else:
            print(f"ERROR! Cannot load from file{os.path.basename(path)}")

    def save(self,path) -> None:
        with open(path, "wb") as f:
            dill.dump(self,f)
            print(f"{self.__class__.__name__} saved to {os.path.basename(path)}")

    def draw(self,path):
        ext = ".jpg"
        if "layer_view" not in os.listdir(path):
            os.mkdir(os.path.join(path,"layer_view"))
        else:
            for png in os.listdir(os.path.join(path,"layer_view")):
                if  png.endswith(ext):
                    os.remove(os.path.join(path,"layer_view",png))
        from math import sqrt
        for i in range(self.num_of_layers):
            n = self.layers[i].number_of_nodes()
            l = int(sqrt(n))*10
            plt.figure(figsize=(l, l))
            nx.draw_networkx(self.layers[i],pos=nx.spring_layout(self.layers[i],k=n**2,scale=n**4))
            plt.savefig(os.path.join(path,"layer_view",f"layer{i}{ext}"))
            plt.clf()
