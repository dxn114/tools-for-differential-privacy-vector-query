import scipy as sp,numpy as np,os,multiprocessing as mp,networkx as nx
from HGraph import HGraph

def x_constraint(x:np.ndarray)->int:
    if x.dtype != int:
        return 0
    n = x.shape[0]

    if (np.sort(x) == np.arange(n,dtype=int)).all():
        return 1
    else:
        return 0

class HREG(HGraph):
    
    def build_layer(self, lc : int):
        layer_size = self.layers[lc].number_of_nodes()
        d = min(layer_size-1, self.M_max)
        reg_graph : nx.Graph = nx.random_regular_graph(d,layer_size)
        x0 = np.arange(layer_size,dtype=int)

        nlc = sp.optimize.NonlinearConstraint(x_constraint,1,1)
        #res = sp.optimize.minimize(self.dist_square_sum,x0,args=(lc,reg_graph),constraints = nlc)
        vector_ids = np.array(self.layers[lc].nodes())[x0]
        for e in reg_graph.edges():
            v1 = vector_ids[e[0]]
            v2 = vector_ids[e[1]]
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
    n.draw(dir_path)
    pass