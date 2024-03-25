import numpy as np,os,networkx as nx
from HGraph import DPHGraph,test_run
from tqdm import trange
import torch

batch_size = 1024
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class ExpHREG(DPHGraph):
    def build_layer(self, lc : int):
        print(f"Building layer {lc}...")
        layer_size = self.layers[lc].number_of_nodes()
        if self.M >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        nodes = list(self.layers[lc].nodes()) # list of vids in layer lc
        k = min(layer_size-1, self.M)
        data = self.data[nodes]       
        data_t = torch.tensor(data,device=device)
        for st in trange(0,layer_size,batch_size):
            end = min(st+batch_size,layer_size)
            batch = torch.tensor(data[st:end],device=device)
            if self.distance=="euclidean":
                dist = torch.cdist(batch,data_t)
            elif self.distance=="cosine":
                dist =  1 - torch.nn.functional.cosine_similarity(batch,data_t,dim=1)
            else:
                raise ValueError("Unsupported distance metric")
            dist = dist.cpu().numpy() 
            dist -= np.random.gumbel(0,1/self.epsilon,size=dist.shape)
            for idx in range(end-st):
                dist[idx,idx+st] = np.inf
            dist = torch.tensor(dist,device=device)
            k_smallest_indices = torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy()
            for idx,v in enumerate(nodes[st:end]):
                edges = [(v,nodes[i]) for i in k_smallest_indices[idx]]
                self.layers[lc].add_edges_from(edges)

if __name__ == '__main__':
    test_run(ExpHREG)
            