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
        d = min(layer_size-1, self.M_max)
        
        nodes = list(self.layers[lc].nodes())
        data = self.data[nodes]
        layer_data = torch.tensor(data,device=torch.device('cuda'))
        gph = knn_graph(layer_data, d,batch=torch.zeros(len(nodes)),loop=False)
        gph = gph.cpu().numpy()
        for u,v in gph.T:
            v1 = nodes[u]
            v2 = nodes[v]
            self.layers[lc].add_edge(v1,v2)


if __name__ == '__main__':
    h = HREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = HREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass