import numpy as np
import os

def gen_randvec_file(*exp):
    data_dir = "randvec"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)   
    dim = 4
    test_filename = "test.npy"
    if not test_filename in data_dir:
        f = open(os.path.join(data_dir, test_filename), "w")
        data_test = np.random.normal(size=(100,dim))
        np.save(os.path.join(data_dir, test_filename),data_test)
        f.close()
    for i in exp:
        dir_path = os.path.join(data_dir,f"10^{i}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = f"randvec_10^{i}.npy"
        if filename in dir_path:
            continue
        filepath : str = os.path.join(dir_path, filename)
        f = open(filepath, "w")
        size = 10**i
        rand_vec = np.random.normal(size=(size,dim))
        np.save(filepath,rand_vec)
        f.close()

def dataset2vec(data_train,data_test,y_train=None,y_test=None,normalize = None):
    data_train = data_train.reshape(data_train.shape[0],-1).astype(float)
    data_test = data_test.reshape(data_test.shape[0],-1).astype(float)
    
    if normalize == "MaxAbs":
        from sklearn.preprocessing import MaxAbsScaler
        data_train=MaxAbsScaler().fit_transform(data_train)
        data_test=MaxAbsScaler().fit_transform(data_test)  
    elif normalize == "Standard":
        from sklearn.preprocessing import StandardScaler
        data_train=StandardScaler().fit_transform(data_train)
        data_test=StandardScaler().fit_transform(data_test)     
    return data_train,data_test,y_train,y_test


def create_datafile(dataset_name,data_train,data_test,y_train=None,y_test=None):
    data_dir = dataset_name
    test_filename = "test.npy"
    if not test_filename in data_dir:
        f = open(os.path.join(data_dir, test_filename), "w")
        choice = np.random.choice(data_test.shape[0], 100, replace=False)
        np.save(os.path.join(data_dir, test_filename),data_test[choice])
        if y_test is not None:
            np.save(os.path.join(data_dir, "y_"+test_filename),y_test[choice])
        f.close()

    # save train data
    data_exp = int(np.log10(data_train.shape[0]))
    for exp in range(3,data_exp+1):
        dir_path = os.path.join(data_dir,f"10^{exp}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = f"{dataset_name}_10^{exp}.npy"
        if filename in dir_path:
            return
        filepath : str = os.path.join(dir_path, filename)

        f = open(filepath, "w")
        if exp==data_exp:
            np.save(filepath,data_train)
            if y_train is not None:
                np.save(os.path.join(dir_path, "y_"+filename),y_train)
        else:
            choice = np.random.choice(data_train.shape[0], int(data_train.shape[0]*(10**(exp-data_exp))), replace=False)
            np.save(filepath,data_train[choice])
            if y_train is not None:
                np.save(os.path.join(dir_path, "y_"+filename),y_train[choice])
        f.close()  

def CIFAR10(extract_features:bool=False)->None:
    from torchvision.datasets import CIFAR10
    dataset_name = "CIFAR10"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_train = CIFAR10(root=data_dir, train=True, download=True).data.swapaxes(1,3).swapaxes(2,3)
    data_test = CIFAR10(root=data_dir, train=False, download=True).data.swapaxes(1,3).swapaxes(2,3)

    data_train,data_test = dataset2vec(data_train,data_test)

    create_datafile(dataset_name,data_train,data_test)      

def MNIST()->None:
    from torchvision.datasets import MNIST
    dataset_name = "MNIST"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_train = MNIST(root=data_dir, train=True, download=True)
    data_test = MNIST(root=data_dir, train=False, download=True)
    y_train = data_train.targets.numpy()
    y_test = data_test.targets.numpy()
    data_train = data_train.data.numpy()
    data_test = data_test.data.numpy()

    data_train,data_test = dataset2vec(data_train,data_test,y_train,y_test)

    create_datafile(dataset_name,data_train,data_test)   

def GloVe():
    dataset_name = "GloVe"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_name = "glove.6B.50d.txt"
    data = []
    with open(os.path.join(data_dir,file_name), "r",encoding="utf-8") as f:
        for line in f:
            line_split = line.split()[1:]
            data.append(line_split)
    data = np.array(data[:-1],dtype=float)
    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, test_size=0.1)
    create_datafile(dataset_name,data_train,data_test)
            
def DEEP():
    dataset_name = "DEEP"
    data_dir = dataset_name
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
    from DEEP.loading import read_fbin
    data=read_fbin(os.path.join(data_dir,"base.10M.fbin"), chunk_size=10**5+10**4)
    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, train_size=10**5, test_size=10**4)
    create_datafile(dataset_name,data_train,data_test)

def SIFT():
    dataset_name = "SIFT"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    import tensorflow_datasets as tfds
    data_train = tfds.load("sift1m",split="database")
    data_test = tfds.load("sift1m",split="test")
    data_test = tfds.as_dataframe(data_test)['embedding']
    data_test = np.stack(data_test.to_numpy())
    data_train = tfds.as_dataframe(data_train)['embedding']
    data_train = np.stack(data_train.to_numpy()) 
    
    create_datafile(dataset_name,data_train,data_test)

if __name__ == "__main__":
    gen_randvec_file(3,4,5,6)
    # MNIST()
    # GloVe()
    # DEEP()
    # SIFT()