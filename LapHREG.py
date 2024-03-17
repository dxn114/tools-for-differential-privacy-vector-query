import os
from HGraph import DPHGraph
import numpy as np
from HREG import HREG

class LapHREG(DPHGraph):
    random_edges = False
    def build(self,M:int,quantize = False,random_edges = False):
        noise = np.random.laplace(0,1/self.epsilon,self.data.shape)
        self.data += noise
        self.random_edges = random_edges
        super().build(M,quantize)
        self.data -= noise

    def build_layer(self, lc : int):
        HREG.build_layer(self,lc)


if __name__ == '__main__':
    class_name = LapHREG.__name__
    
    dir_path = os.path.join(f"randvec_{class_name}","10^3") 
    npy_path = os.path.join(dir_path,"randvec128_10^3.npy") 
    h = LapHREG(path=npy_path)
    h.build(16)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = LapHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass