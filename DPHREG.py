import numpy as np,os,networkx as nx
from HGraph import HGraph
# from sklearn.metrics import pairwise_distances
# from joblib import Parallel,delayed
from tqdm import trange
import torch

# def select_neighbors(data,nodes,idx,k,epsilon):
#     dist_v = pairwise_distances(data[idx].reshape(1,-1),data).ravel()
#     dist_v = np.delete(dist_v,idx) - np.random.gumbel(0,1/epsilon,data.shape[0]-1)
#     k_smallest_indices = np.argpartition(dist_v,k)[:k]
#     adj_nodes = [nodes[idx] for idx in k_smallest_indices]
#     return adj_nodes
class DPHREG(HGraph):
    epsilon : float = 0
    def __init__(self,epsilon=0.5) -> None:
        super().__init__()
        self.epsilon = epsilon
    def build_layer(self, lc : int):
        print(f"Building layer {lc}...")
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        nodes = list(self.layers[lc].nodes()) # list of vids in layer lc
        k = min(layer_size-1, self.M_max)
        data = self.data[nodes]       
        # neighbors_list = Parallel(n_jobs=15)(delayed(select_neighbors)(data,nodes,idx,k,self.epsilon) for idx in trange(layer_size))
        # for idx,v in enumerate(nodes):
        #     # dist_v = pairwise_distances(data[idx].reshape(1,-1),data).ravel() + np.random.gumbel(0,1/self.epsilon,layer_size)
        #     # k_smallest_indices = np.argpartition(dist_v,k)[:k]
        #     adj_nodes = neighbors_list[idx]
        #     for u in adj_nodes:
        #         self.layers[lc].add_edge(u,v)
        data_t = torch.tensor(data,device=torch.device('cuda'))
        batch_size = 1024
        for st in trange(0,layer_size,batch_size):
            end = min(st+batch_size,layer_size)
            batch = torch.tensor(data[st:end],device=torch.device('cuda'))
            dist = torch.cdist(batch,data_t)
            dist = dist.cpu().numpy() 
            new_dist = np.zeros((dist.shape[0],dist.shape[1]-1))
            for idx in range(end-st):
                new_dist[idx] = np.delete(dist[idx],idx+st)
            dist = new_dist - np.random.gumbel(0,1/self.epsilon,size=new_dist.shape)
            dist = torch.tensor(dist,device=torch.device('cuda'))
            k_smallest_indices = torch.topk(dist,k,largest=False,sorted=False).indices.cpu().numpy()
            # k_smallest_indices = np.argpartition(dist,kth=k,axis=1)[:,:k]
            for idx,v in enumerate(nodes[st:end]):
                edges = [(v,nodes[i]) for i in k_smallest_indices[idx]]
                self.layers[lc].add_edges_from(edges)


if __name__ == '__main__':
    h = DPHREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^4") 
    csv_path = os.path.join(dir_path,"randvec128_10^4.npy") 
    h_path = csv_path.replace(".npy",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = DPHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass
            
            