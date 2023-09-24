import numpy as np,scipy as sp,os,multiprocessing as mp,networkx as nx
from HGraph import HGraph
from sklearn.neighbors import kneighbors_graph

class HREG(HGraph):
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        if self.M_max >= layer_size-1:
            self.layers[lc] = nx.complete_graph(self.layers[lc].nodes())
            return
        d = min(layer_size-1, self.M_max)
        
        layer_data = self.data[self.layers[lc].nodes()]
        Adj_Mat = kneighbors_graph(layer_data,d,mode='distance',include_self=False)
        nodelist = list(self.layers[lc].nodes())
        for u in range(layer_size):
            for v in range(u+1,layer_size):
                if Adj_Mat[u,v] == 0:
                    continue
                v1 = nodelist[u]
                v2 = nodelist[v]
                self.layers[lc].add_edge(v1,v2,weight=Adj_Mat[u,v])


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