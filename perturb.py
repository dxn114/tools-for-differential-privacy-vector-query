import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tqdm import trange
t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp = 4
data = torch.randint(0,256,size=(10**exp,128),device=device,dtype=float)
pw_dist = torch.cdist(data,data).to('cpu').numpy()
def perturb_i(i,idx,pw_dist_new):
    # f, g for all v in the ith perturbation
    fgi = np.zeros((data.shape[0],2),dtype=int)
    for v in range(data.shape[0]):
        if v == idx:
            continue
        old_dist_v = pw_dist[idx,v]
        new_dist_v = pw_dist_new[i,v]
        f = (pw_dist[v]<old_dist_v).sum()
        g = (pw_dist[v]<new_dist_v).sum()
        if old_dist_v<new_dist_v: 
            g-=1
        if f > g:
            f,g = g,f
        fgi[v] = f,g
    return fgi
def perturb(indices,new,K):
    perturb_time = new.shape[0]
    freqK = np.zeros((len(K),perturb_time),dtype=int)
    pw_dist_new = torch.cdist(new,data).to('cpu').numpy()
    # fg[i] is the f,g for all v in the ith perturbation
    fg = Parallel(n_jobs=-1)(delayed(perturb_i)(i,indices[i],pw_dist_new) for i in trange(perturb_time))
    for i in range(perturb_time):
        for v in range(data.shape[0]):
            if v == indices[i]:
                continue
            f,g = fg[i][v]
            for j,k in enumerate(K):
                if f<=k and g>k:
                    freqK[j,i]+=1    
    return freqK

def one_perturb(k=None):
    freq100 = []
    for t in range(10):
        new = np.random.randint(0,256,size=128)
        freq = perturb(0,new)
        freq100.append(freq[100])
        plt.plot(freq)
    plt.legend()
    plt.savefig(f"randvec/10^{exp}/perturb1.png")
    plt.close()

def random_perturb(perturb_time=10000):
    K = [10,20,50,100]
    new = torch.randint(0,256,size=(perturb_time,128),device=device,dtype=float)
    indices = np.random.randint(0,10**exp,size=(perturb_time,))
    freqK = perturb(indices,new,K)
    mu = np.mean(freqK,axis=1)
    for i,freqk in enumerate(freqK):
        k = K[i]
        plt.hist(freqk,bins=data.shape[0]//10,histtype='step',label=f"$k={k}$")
        # plt.axvline(mu[i],color='r',label=f"$\mu={mu[i]}$")
        # plt.axvline(k,color='k',label=f"$k={k}$")
    plt.xlabel(f"$X_k$")
    plt.ylabel(f"$Freq$")
    plt.title(f"k = {K}")
    plt.legend()
    plt.savefig(f"randvec/10^{exp}/random_perturb_freqk.png")

random_perturb(perturb_time=3000)
t = time.time()-t
print(f"Time taken: {t} seconds")