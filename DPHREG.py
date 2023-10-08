import numpy as np,os,networkx as nx
from HGraph import HGraph
import torch
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
class DPHREG(HGraph):
    epsilon : float = 0
    delta : float = 0
    def __init__(self,epsilon=0.2,delta=0.05) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
    def build_layer(self, lc : int):
        # privately select top-d
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        nodes = list(self.layers[lc].nodes()) # list of vids in layer lc
        m = layer_size-1
        k = min(layer_size-1, self.M_max)
        for v in self.layers[lc].nodes():
            dist_vec = np.zeros(0) # distances from c to all other nodes in nodes
            if self.Dist_Mat.size != 0:
                dist_vec = self.Dist_Mat[v][nodes]
            else:
                #dist_vec = pairwise_distances(self.data[v].reshape(1,-1),self.data[nodes],metric='euclidean',n_jobs=-1).ravel()
                v_tensor = torch.from_numpy(self.data[v].reshape(1,-1)).to(device)
                nodes_tensor = torch.from_numpy(self.data[nodes]).to(device)
                dist_vec = torch.pairwise_distance(v_tensor,nodes_tensor,p=2).ravel().to("cpu").numpy()

            sorted_dist_vec = np.sort(dist_vec)

            s_f = 0
            for i in range(0,m):
                s_f = max(abs(sorted_dist_vec[i+1] - sorted_dist_vec[i]),s_f)

            lambda_os = 8*s_f*np.sqrt(k*np.log(m/self.delta))/self.epsilon

            noisy_dist_vec = dist_vec + np.random.laplace(0,lambda_os,size=dist_vec.shape)
            
            k_smallest_indices = np.argpartition(noisy_dist_vec,k)[:k]
            adj_nodes = [nodes[idx] for idx in k_smallest_indices]
            for u in adj_nodes:
                self.layers[lc].add_edge(u,v)

if __name__ == '__main__':
    h = DPHREG()
    dir_path = os.path.join(f"randvec_{h.__class__.__name__}","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    h_path = csv_path.replace(".csv",f".{h.__class__.__name__.lower()}")
    h.build(csv_path,16)
    h.save(h_path)
    n = DPHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass
            
            