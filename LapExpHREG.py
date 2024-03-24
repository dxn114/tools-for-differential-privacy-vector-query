import os,networkx as nx
from HGraph import DPHGraph
import numpy as np
from HREG import HREG
import torch
from torch_cluster import knn_graph
batch_size = 1024
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def Exp_KNN(q,data,k,epsilon): # Assume q and data are disjoint
    q = torch.tensor(q,device=device)
    data_t = torch.tensor(data,device=device)
    dist = torch.cdist(q,data_t)
    dist = dist.cpu().numpy() 
    dist -= np.random.gumbel(0,1/epsilon,size=dist.shape)
    dist = torch.tensor(dist,device=device)
    k_smallest_indices = torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy()
    return k_smallest_indices

class LapExpHREG(DPHGraph):
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        if self.M >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        k = min(layer_size-1, self.M)
        nodes_up = set(self.layers[lc+1].nodes()) if lc+1 < self.num_of_layers else set()
        nodes_this = set(self.layers[lc].nodes()) - nodes_up
        nodes_this = list(nodes_this)
        nodes_up = list(nodes_up)
        if len(nodes_this) <= k:
            for u in nodes_up:
                self.layers[lc].add_edges_from([(u,v) for v in nodes_this])
            return
        data_up = self.data[nodes_up]
        data_this = self.data[nodes_this]
        noisy_data_this = data_this + np.random.laplace(0,1/self.epsilon,data_this.shape)
        gph = knn_graph(torch.tensor(noisy_data_this,device=device).float(),k,loop=False)
        gph = gph.cpu().numpy().T
        self.layers[lc].add_edges_from([(nodes_this[i],nodes_this[j]) for i,j in gph])
        batch_size = 1024
        for i in range(0,data_up.shape[0],batch_size):
            step = batch_size if i+batch_size < data_up.shape[0] else data_up.shape[0] - i
            q = data_up[i:i+step]
            k_smallest_indices = Exp_KNN(q,data_this,k,self.epsilon)
            for j,neighbors in enumerate(k_smallest_indices):
                source = nodes_up[i+j]
                edges = [(source,nodes_this[n]) for n in neighbors]
                self.layers[lc].add_edges_from(edges)

if __name__ == '__main__':
    class_name = LapExpHREG.__name__
    
    dir_path = os.path.join(f"randvec","10^3") 
    npy_path = os.path.join(dir_path,"randvec_10^3.npy") 
    h = LapExpHREG(path=npy_path)
    h.build(16)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = LapExpHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass