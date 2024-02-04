import os,networkx as nx
from HGraph import DPHGraph
import numpy as np
from torch_cluster import knn_graph
import torch

class LapHREG(DPHGraph):
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        k = min(layer_size-1, self.M_max)
        
        nodes = list(self.layers[lc].nodes())
        data = self.data[nodes]
        data += np.random.laplace(0,1/self.epsilon,data.shape)
        data_t = torch.tensor(data,device=torch.device('cuda'))
        gph = knn_graph(data_t, k,batch=torch.zeros(len(nodes)),loop=False)
        gph = gph.cpu().numpy()
        edges = [(nodes[u],nodes[v]) for u,v in gph.T if u!=v]
        self.layers[lc].add_edges_from(edges)


if __name__ == '__main__':
    h = LapHREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = LapHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass