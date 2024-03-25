import os,networkx as nx
from HGraph import HGraph, test_run
from torch_cluster import knn_graph
import torch

class HREG(HGraph):
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        if self.M >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        k = min(layer_size-1, self.M)
        
        nodes = list(self.layers[lc].nodes())
        data = self.data[nodes]
        layer_data = torch.tensor(data,device=torch.device('cuda'))
        gph = knn_graph(layer_data, k,batch=torch.zeros(len(nodes)),loop=False,cosine=True if self.distance=="cosine" else False)
        gph = gph.cpu().numpy().T
        self.layers[lc].add_edges_from([(nodes[i],nodes[j]) for i,j in gph])

if __name__ == '__main__':
    test_run(HREG)