import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tqdm import trange
import os
t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp = 3
dim = 128
K = [10,20]
data_type = ''
data = torch.Tensor([])
pw_dist = torch.Tensor([])
pw_dist_new = torch.Tensor([])
low = 0
high = 1024

def perturb_i(i,idx):
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
    global pw_dist,pw_dist_new
    pw_dist = torch.cdist(data,data).to('cpu').numpy()
    pw_dist_new = torch.cdist(new,data).to('cpu').numpy()
    # fg[i] is the f,g for all v in the ith perturbation
    fg = Parallel(n_jobs=10)(delayed(perturb_i)(i,indices[i]) for i in trange(perturb_time))
    for i in range(perturb_time):
        for v in range(data.shape[0]):
            if v == indices[i]:
                continue
            f,g = fg[i][v]
            for j,k in enumerate(K):
                if f<=k and g>k:
                    freqK[j,i]+=1    
    return freqK

def random_perturb_int(perturb_time=10000):
    if os.path.exists(f"randvec/10^{exp}/random_perturb_int_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/random_perturb_int_freqK.npy")
    else:
        global data,data_type
        if data_type != 'int':
            data = torch.randint(low,high,size=(10**exp,dim),device=device,dtype=float)
            data_type = 'int'
        new = torch.randint(low,high,size=(perturb_time,dim),device=device,dtype=float)
        indices = np.random.randint(0,10**exp,size=(perturb_time,))
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/random_perturb_int_freqK.npy",freqK)

    hist(freqK,f"randvec/10^{exp}/random_perturb_int_freqk.png")

def random_perturb_int_8d(perturb_time=10000):
    if os.path.exists(f"randvec/10^{exp}/random_perturb_int_8d_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/random_perturb_int_8d_freqK.npy")
    else:
        global data,data_type
        if data_type != 'int':
            data = torch.randint(low,high,size=(10**exp,dim),device=device,dtype=float)
            data_type = 'int'
        new_dims = torch.randint(low,high,size=(perturb_time,8),device=device,dtype=float)
        indices = np.random.randint(0,10**exp,size=(perturb_time,))
        new = data[indices]
        new[:,:8] = new_dims
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/random_perturb_int_8d_freqK.npy",freqK)

    hist(freqK,f"randvec/10^{exp}/random_perturb_int_8d_freqk.png")

def random_perturb_01(perturb_time=10000):
    if os.path.exists(f"randvec/10^{exp}/random_perturb_01_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/random_perturb_01_freqK.npy")
    else:
        global data,data_type
        if data_type != '01':
            data = torch.ones((10**exp,dim),device=device,dtype=float)*(1/2)
            data = torch.bernoulli(data).to(device)
            data_type = '01'
        
        new = torch.ones((perturb_time,dim),device=device,dtype=float)*(1/2)
        new = torch.bernoulli(new).to(device)
        indices = np.random.randint(0,10**exp,size=(perturb_time,))
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/random_perturb_01_freqK.npy",freqK)

    hist(freqK,f"randvec/10^{exp}/random_perturb_01_freqk.png")

def random_perturb_01_8d(perturb_time=10000):
    if os.path.exists(f"randvec/10^{exp}/random_perturb_01_8d_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/random_perturb_01_8d_freqK.npy")
    else:
        global data,data_type
        if data_type != '01':
            data = torch.ones((10**exp,dim),device=device,dtype=float)*(1/2)
            data = torch.bernoulli(data).to(device)
            data_type = '01'

        new_dims = torch.ones((perturb_time,8),device=device,dtype=float)*(1/2)
        new_dims = torch.bernoulli(new_dims).to(device)
        indices = np.random.randint(0,10**exp,size=(perturb_time,))
        new = data[indices]
        new[:,:8] = new_dims
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/random_perturb_01_8d_freqK.npy",freqK)
    
    hist(freqK,f"randvec/10^{exp}/random_perturb_01_8d_freqk.png")

def hist(freqK,path):
    mu = np.mean(freqK,axis=1)
    for i,freqk in enumerate(freqK):
        k = K[i]
        plt.hist(freqk,bins=100,histtype='step',label=f"$k={k}$")
        plt.axvline(mu[i],color = 'k',label=f"$\mu_{{k={k}}}={mu[i]}$")
        # plt.axvline(k,label=f"$k={k}$")
    plt.xlabel(f"$X_k$")
    plt.ylabel(f"$Freq$")
    plt.title(f"k = {K}")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot(freqK,path):
    plt.figure(figsize=(30,10))
    for i,freqk in enumerate(freqK):
        k = K[i]
        plt.plot(freqk,label=f"$k={k}$")
    plt.xlabel(f"$dist$")
    plt.ylabel(f"$X_k$")
    plt.title(f"k = {K}")
    plt.legend()
    plt.savefig(path)
    plt.close()

def dist_perturb_int(perturb_time=1000):
    if os.path.exists(f"randvec/10^{exp}/dist_perturb_int_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/dist_perturb_int_freqK.npy")
    else:
        global data,data_type
        if data_type != 'int':
            data = torch.randint(low,high,size=(10**exp,dim),device=device,dtype=float)
            data_type = 'int'
        new = torch.randint(low,high,size=(perturb_time,dim),device=device,dtype=float)
        idx = np.random.randint(0,10**exp)
        indices = idx*np.ones((perturb_time,),dtype=int)
        for i ,v in enumerate(new):
            new[i] = data[idx]+(v/torch.norm(v))*i
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/dist_perturb_int_freqK.npy",freqK)

    plot(freqK,f"randvec/10^{exp}/dist_perturb_int_freqk.png")

def dist_perturb_fix_direction_int(perturb_time=1000):
    if os.path.exists(f"randvec/10^{exp}/dist_perturb_fix_direction_int_freqK.npy"):
        freqK = np.load(f"randvec/10^{exp}/dist_perturb_fix_direction_int_freqK.npy")
    else:
        global data,data_type
        if data_type != 'int':
            data = torch.randint(low,high,size=(10**exp,dim),device=device,dtype=float)
            data_type = 'int'
        new = torch.zeros(size=(perturb_time,dim),device=device,dtype=float)
        delta = torch.randint(low,high,size=(dim,),device=device,dtype=float)
        delta = delta/torch.norm(delta)
        idx = np.random.randint(0,10**exp)
        indices = idx*np.ones((perturb_time,),dtype=int)
        for i in range(perturb_time):
            new[i] = data[idx]+i*delta
        freqK = perturb(indices,new,K)
        np.save(f"randvec/10^{exp}/dist_perturb_fix_direction_int_freqK.npy",freqK)

    plot(freqK,f"randvec/10^{exp}/dist_perturb_fix_direction_int_freqk.png")
# random_perturb_int()
# random_perturb_int_8d()
# random_perturb_01()
# random_perturb_01_8d()
dist_perturb_int()
dist_perturb_fix_direction_int()
t = time.time()-t
print(f"Time taken: {t} seconds")