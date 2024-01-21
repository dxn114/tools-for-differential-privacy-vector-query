import numpy as np,os,networkx as nx
from HGraph import HGraph
from sklearn.metrics import pairwise_distances
from joblib import Parallel,delayed
from tqdm import trange
def select_neighbors(data,nodes,idx,k,epsilon):
    vec = data[idx].reshape(1,-1)
    data = np.delete(data,idx,axis=0)
    dist_v = pairwise_distances(vec,data).ravel() - np.random.gumbel(0,1/epsilon,data.shape[0])
    k_smallest_indices = np.argpartition(dist_v,k)[:k]
    adj_nodes = [nodes[idx] for idx in k_smallest_indices]
    return adj_nodes
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
        data = self.data[nodes]
        k = min(layer_size-1, self.M_max)
        neighbors_list = Parallel(n_jobs=14)(delayed(select_neighbors)(data,nodes,idx,k,self.epsilon) for idx in trange(layer_size))
        for idx,v in enumerate(nodes):
            # dist_v = pairwise_distances(data[idx].reshape(1,-1),data).ravel() + np.random.gumbel(0,1/self.epsilon,layer_size)
            # k_smallest_indices = np.argpartition(dist_v,k)[:k]
            adj_nodes = neighbors_list[idx]
            for u in adj_nodes:
                self.layers[lc].add_edge(u,v)

if __name__ == '__main__':
    h = DPHREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^4") 
    csv_path = os.path.join(dir_path,"randvec128_10^4.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = DPHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass
            
            