import os,networkx as nx
from HGraph import HGraph
from torch_cluster import knn_graph
import torch

class HREG(HGraph):
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        k = min(layer_size-1, self.M_max)
        
        nodes = list(self.layers[lc].nodes())
        data = self.data[nodes]
        layer_data = torch.tensor(data,device=torch.device('cuda'))
        gph = knn_graph(layer_data, k,batch=torch.zeros(len(nodes)),loop=False)
        gph = gph.cpu().numpy()
        edges = [(nodes[u],nodes[v]) for u,v in gph.T if u!=v]
        self.layers[lc].add_edges_from(edges)

if __name__ == '__main__':
    class_name = HREG.__name__
    
    dir_path = os.path.join(f"randvec_{class_name}","10^3") 
    npy_path = os.path.join(dir_path,"randvec128_10^3.npy") 
    h = HREG(path=npy_path)
    h.build(8)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = HREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass