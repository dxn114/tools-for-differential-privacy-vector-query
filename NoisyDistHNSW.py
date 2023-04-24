from HNSW import HNSW
import dill,os,numpy as np
class NoisyDistHNSW(HNSW):
    sigma = 0

    def set_Noise_dev(self, sigma):
        self.sigma = sigma
    
    def dist(self,q1,q2):
        return np.sum(np.square(q1 - q2))+np.random.normal(0,self.sigma)

    def load(self,path:str):
        print(f"Loading HNSW from {os.path.basename(path)} ...")
        if(path.endswith(".hnsw") or path.endswith(".hnswh")):
            with open (path,"rb") as f:
                m :NoisyDistHNSW = dill.load(f)
                self.layers = m.layers
                self.data = m.data
                self.ep  = m.ep
                self.next_seqno = m.next_seqno
                self.num_of_layers = m.num_of_layers
                self.dist_type = m.dist_type
                self.algchoose = m.algchoose
                self.sigma = m.sigma

            heu = ""    
            if self.algchoose == 4:
                heu = " (with heuristic neighbor selection)"
            print(f"File {os.path.basename(path)} loaded as HNSW{heu}")
        else:
            print(f"ERROR! Cannot load from file{os.path.basename(path)}")