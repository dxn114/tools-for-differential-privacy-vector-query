import os
from HGraph import DPHGraph
import numpy as np
from HREG import HREG

class LapHREG(DPHGraph):
    def build(self,M:int):
        noise = np.random.laplace(0,1/self.epsilon,self.data.shape)
        self.data += noise
        super().build(M)
        self.data -= noise

    def build_layer(self, lc : int):
        HREG.build_layer(self,lc)


if __name__ == '__main__':
    class_name = LapHREG.__name__
    
    dir_path = os.path.join(f"randvec","10^3") 
    npy_path = os.path.join(dir_path,"randvec_10^3.npy") 
    h = LapHREG(path=npy_path)
    h.build(16)
    h_path = npy_path.replace(".npy",f".{class_name.lower()}")
    h.save(h_path)
    n = LapHREG()
    n.load(h_path)
    #n.draw(dir_path)
    pass