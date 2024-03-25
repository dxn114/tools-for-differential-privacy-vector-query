import os
from HGraph import DPHGraph, test_run
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
    test_run(LapHREG)