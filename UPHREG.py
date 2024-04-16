from HGraph import HGraph, DPHGraph, DPknn,test_run
from HREG import HREG
import numpy as np
import networkx as nx
import torch
import time
from torch_cluster import knn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UPHREG(HGraph):
    num_protected : int = 0

    def build_layer(self, lc):
        HREG.build_layer(self,lc)

    def build(self,M:int,distance:str="euclidean",num_unprotected : int = None):
        # empty pseudo vectors are those who has no real neighbors
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            self.M = M
            self.M_max = 2*M        
            self.distance = distance

            if num_unprotected == None: 
                num_unprotected = int(self.num_of_vectors/2)
            self.num_protected = self.num_of_vectors - num_unprotected

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
            k_unprotected_nbr = knn(unprotected_data_t,protected_data_t,self.M,cosine= True if self.distance=="cosine" else False)
            k_unprotected_nbr[0] += num_unprotected
            k_unprotected_nbr = k_unprotected_nbr.cpu().numpy().T

            self.layers[0].add_nodes_from(range(num_unprotected,self.num_of_vectors))
            self.layers[0].add_edges_from(k_unprotected_nbr)

            t = time.time()-t
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

class ExpUPHREG(DPHGraph):
    num_protected : int = 0

    def build_layer(self, lc):
        HREG.build_layer(self,lc)

    def build(self,M:int,distance:str="euclidean",num_unprotected : int = None):
        # empty pseudo vectors are those who has no real neighbors
        class_name = self.__class__.__name__
        if(self.data.size>0):
            print(f"Building {class_name} from {self.data_file} ...")
            t = time.time()
            mL:float = 1/(np.log(M))
            self.M = M
            self.M_max = 2*M        
            self.distance = distance

            if num_unprotected == None: 
                num_unprotected = int(self.num_of_vectors/2)
            self.num_protected = self.num_of_vectors - num_unprotected

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
            k_unprotected_nbr = DPknn(protected_data_t,unprotected_data_t,self.epsilon,self.M)

            self.layers[0].add_nodes_from(range(num_unprotected,self.num_of_vectors))
            for i, knbrs in enumerate(k_unprotected_nbr):
                self.layers[0].add_edges_from([(i+num_unprotected,nbr) for nbr in knbrs])

            t = time.time()-t
            print(f"{class_name} from data file {self.data_file} built in {t:.3f} seconds.")
        else: 
            print(f"ERROR! No data to build {class_name} from.")

class LapUPHREG(DPHGraph):
    num_protected : int = 0
    def build(self,M:int, distance:str="euclidean", num_unprotected : int = None):
        if num_unprotected == None: 
            num_unprotected = int(self.num_of_vectors/2)
        self.num_protected = self.num_of_vectors - num_unprotected      

        noise = np.random.laplace(0,np.sqrt(self.data.shape[1])/self.epsilon,(self.num_protected,self.data.shape[1]))

        self.data[:num_unprotected] += noise
        super().build(M,distance)
        self.data[:num_unprotected] -= noise

    def build_layer(self, lc : int):
        HREG.build_layer(self,lc)

if __name__ == "__main__":
    test_run(UPHREG)
    test_run(ExpUPHREG)