import scipy as sp,numpy as np,os,multiprocessing as mp,networkx as nx
from HGraph import HGraph

#reorder M according to x
def reorder(x : np.ndarray,M:np.ndarray)->np.ndarray:
    n = x.shape[0]
    M_ : np.ndarray = np.empty(M.shape,dtype=np.ndarray)
    for i in range(n):
        if M.ndim == 1:
            M_[int(x[i])]=M[i]
        elif M.ndim == 2:
            M_[int(x[i]),:]=M[i,:]
    return M_

def x_constraint(x:np.ndarray)->int:
    if x.dtype != int:
        return 0
    n = x.shape[0]

    if (np.sort(x) == np.arange(n,dtype=int)).all():
        return 1
    else:
        return 0

#sum of squared pairwise distances
def dist_square_sum(x : np.ndarray,*args)->float:
    data : np.ndarray = args[0]
    A : nx.Graph = args[1]
    size = data.shape[0]
    pem = reorder(x,data)

    F = np.ones((size,size),dtype=int)
    for i in range(size):
        d2 = int(np.inner(pem[i],pem[i]))
        F[i] *= d2
    G = F + np.transpose(F) - 2*(pem @ (np.transpose(pem)))
    
    H = 0
    for e in A.edges():
        H += G[e[0],e[1]]       
    return H

class HREG(HGraph):
    def build_layer(self, lc):
        layer_size = self.layers[lc].number_of_nodes()
        d = min(layer_size-1, self.M_max)
        reg_graph = nx.random_regular_graph(d,layer_size)
        x0 = np.arange(layer_size,dtype=int)
        layer_data = self.data[list(self.layers[lc].nodes()),:]
        nlc = sp.optimize.NonlinearConstraint(x_constraint,1,1)
        #res = sp.optimize.minimize(dist_square_sum,x0,args=(layer_data,reg_graph),constraints = nlc)
        vector_ids = reorder(x0,np.array(list(self.layers[lc].nodes())))
        for e in reg_graph.edges():
            v1 = vector_ids[e[0]]
            v2 = vector_ids[e[1]]
            self.layers[lc].add_edge(v1,v2)

if __name__ == '__main__':
    hreg = HREG()
    dir_path = os.path.join("randvec_HREG","10^3") 
    csv_path = os.path.join(dir_path,"randvec128_10^3.csv") 
    hreg_path = csv_path.replace(".csv",".hreg")
    hreg.build(csv_path,16)
    #hreg.draw(dir_path)
    hreg.save(hreg_path)
    nhreg = HREG()
    nhreg.load(hreg_path)
    pass