import numpy as np,os,networkx as nx
from HGraph import DPHGraph
from tqdm import trange
import torch

batch_size = 1024
def DP_KNNG(data,k,epsilon):
    G = nx.Graph()
    size = data.shape[0]
    data_t = torch.tensor(data,device=torch.device('cuda'))
    for st in trange(0,size,batch_size):
        end = min(st+batch_size,size)
        batch = torch.tensor(data[st:end],device=torch.device('cuda'))
        dist = torch.cdist(batch,data_t)
        dist = dist.cpu().numpy() 
        new_dist = np.zeros((dist.shape[0],dist.shape[1]-1))
        for idx in range(end-st):
            new_dist[idx] = np.delete(dist[idx],idx+st)
        dist = new_dist - np.random.gumbel(0,1/epsilon,size=new_dist.shape)
        dist = torch.tensor(dist,device=torch.device('cuda'))
        k_smallest_indices = torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy()
        for i,neighbors in enumerate(k_smallest_indices):
            G.add_edges_from([(i+st,neighbor) for neighbor in neighbors])
        
class ExpHREG(DPHGraph):
    def build_layer(self, lc : int):
        print(f"Building layer {lc}...")
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        nodes = list(self.layers[lc].nodes()) # list of vids in layer lc
        k = min(layer_size-1, self.M_max)
        data = self.data[nodes]       
        data_t = torch.tensor(data,device=torch.device('cuda'))
        for st in trange(0,layer_size,batch_size):
            end = min(st+batch_size,layer_size)
            batch = torch.tensor(data[st:end],device=torch.device('cuda'))
            dist = torch.cdist(batch,data_t)
            dist = dist.cpu().numpy() 
            dist -= np.random.gumbel(0,1/self.epsilon,size=dist.shape)
            for idx in range(end-st):
                dist[idx,idx+st] = np.inf
            dist = torch.tensor(dist,device=torch.device('cuda'))
            k_smallest_indices = torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy()
            for idx,v in enumerate(nodes[st:end]):
                edges = [(v,nodes[i]) for i in k_smallest_indices[idx]]
                self.layers[lc].add_edges_from(edges)

if __name__ == '__main__':
    h = ExpHREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^4") 
    csv_path = os.path.join(dir_path,"randvec128_10^4.npy") 
    h_path = csv_path.replace(".npy",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = ExpHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass
            
            