from HGraph import HGraph, DPUPHGraph, DPknn,test_run
from HKNN import HKNN
import numpy as np
import networkx as nx
import torch
import time
from torch_cluster import knn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UPHKNN(HGraph):
    num_protected : int = 0

    def __init__(self, path:str = None, path2:str = None, num_unprotected = None) -> None:
        if path == None:
            super().__init__(None)
        else:
            if path2 == None:
                super().__init__(path)            
                if num_unprotected == None or num_unprotected < 0: 
                    num_unprotected = int(self.num_of_vectors/2)
                elif num_unprotected < 1:
                    num_unprotected = int(self.num_of_vectors*num_unprotected)
                self.num_protected = self.num_of_vectors - num_unprotected
            else:
                self.layers = []
                import os
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

    def build_layer(self, lc):
        HKNN.build_layer(self,lc)

    def build(self,M:int,cosine:bool = False):
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            self.M = M
            self.M_max = 2*M        
            self.cosine = cosine
            num_unprotected = self.num_of_vectors - self.num_protected

            # unprotected data are indexed lower than protected data

            l = (-np.log(np.random.rand(num_unprotected))*mL).astype(int)# new element’s level (count from 0)
            for i in range(num_unprotected):
                while len(self.layers)-1 < l[i]:
                    self.layers.append(nx.Graph())
                for j in range(l[i]+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            self.ep = int(list(self.layers[-1].nodes())[-1])
            
            for lc in range(self.num_of_layers-1,-1,-1):
                self.build_layer(lc)

            self.layers[0].add_nodes_from(range(num_unprotected,self.num_of_vectors))
            
            unprotected_data_t = torch.tensor(self.data[:num_unprotected],dtype=float,device=device)
            protected_data_t = torch.tensor(self.data[num_unprotected:],dtype=float,device=device)
            k_unprotected_nbr = knn(unprotected_data_t,protected_data_t,self.M,cosine=self.cosine)
            k_unprotected_nbr[0] += num_unprotected
            k_unprotected_nbr = k_unprotected_nbr.cpu().numpy().T

            self.layers[0].add_nodes_from(range(num_unprotected,self.num_of_vectors))
            self.layers[0].add_edges_from(k_unprotected_nbr)

            t = time.time()-t
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

class ExpUPHKNN(DPUPHGraph):

    def build_layer(self, lc):
        HKNN.build_layer(self,lc)

    def build(self,M:int,cosine:bool = False):

        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            self.M = M
            self.M_max = 2*M        
            self.cosine = cosine

            num_unprotected = self.num_of_vectors - self.num_protected

            # unprotected data are indexed lower than protected data

            l = (-np.log(np.random.rand(self.num_of_vectors))*mL).astype(int)# new element’s level (count from 0)
            for i in range(num_unprotected):
                while len(self.layers)-1 < l[i]:
                    self.layers.append(nx.Graph())
                for j in range(l[i]+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            self.ep = int(list(self.layers[-1].nodes())[-1])
            
            for lc in range(self.num_of_layers-1,-1,-1):
                self.build_layer(lc)

            self.layers[0].add_nodes_from(range(num_unprotected,self.num_of_vectors))
            
            unprotected_data_t = torch.tensor(self.data[:num_unprotected],dtype=float,device=device)
            protected_data_t = torch.tensor(self.data[num_unprotected:],dtype=float,device=device)
            k_unprotected_nbr = DPknn(protected_data_t,unprotected_data_t,self.epsilon,self.M,cosine=self.cosine)

            for i in range(num_unprotected,self.num_of_vectors):
                while len(self.layers)-1 < l[i]:
                    self.layers.append(nx.Graph())
                for j in range(l[i]+1):
                    self.layers[j].add_node(i)

            self.num_of_layers = len(self.layers)
            self.ep = int(list(self.layers[-1].nodes())[-1])

            for i, knbrs in enumerate(k_unprotected_nbr):
                edges = [(i+num_unprotected,nbr) for nbr in knbrs]
                self.layers[0].add_edges_from(edges)
                for lc in range(1,self.num_of_layers):
                    if self.layers[lc].number_of_nodes() <= self.M + 1:
                        self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
                    else:
                        edges = [(u,v) for u,v in edges if lc <= l[u] and lc <= l[v]]
                        self.layers[lc].add_edges_from(edges)

            t = time.time()-t
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

class LapUPHKNN(DPUPHGraph):
    def build(self,M:int, cosine = False):   
        num_unprotected = self.num_of_vectors - self.num_protected
        
        noise = np.random.laplace(0,np.sqrt(self.data.shape[1])/self.epsilon,(self.num_protected,self.data.shape[1]))

        self.data[num_unprotected:] += noise
        super().build(M,cosine)
        self.data[num_unprotected:] -= noise

    def build_layer(self, lc : int):
        HKNN.build_layer(self,lc)

if __name__ == "__main__":
    test_run(UPHKNN)
    test_run(ExpUPHKNN)