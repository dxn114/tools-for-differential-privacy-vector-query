import numpy as np
import os,torch

def gen_randvec_file(*exp):
    data_dir = "randvec"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)   
    dim = 128
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

def dataset2vec(data_train,data_test,extract_features:bool=False):
    
    if extract_features:
        from torchvision.models import resnet18
        from torch.utils.data import DataLoader
        from torch import nn
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_model = resnet18(pretrained=True)
        feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1]).to(device)
        feature_extractor.eval()
        data_train = torch.tensor(data_train,dtype=torch.float32).to(device)
        data_test = torch.tensor(data_test,dtype=torch.float32).to(device)
        train_dataloader = DataLoader(data_train, batch_size=128, shuffle=False)
        test_dataloader = DataLoader(data_test, batch_size=128, shuffle=False)
        train_features = np.array([])
        test_features = np.array([])
        for X in train_dataloader:
            feature_vec = feature_extractor(X)
            feature_vec = feature_vec.cpu().detach().numpy()
            if train_features.size == 0:
                train_features = feature_vec
            else:
                train_features = np.concatenate((train_features, feature_vec), axis=0)
        for X in test_dataloader:
            feature_vec = feature_extractor(X)
            feature_vec = feature_vec.cpu().detach().numpy()
            if test_features.size == 0:
                test_features = feature_vec
            else:
                test_features = np.concatenate((test_features, feature_vec), axis=0)
        data_train = train_features
        data_test = test_features

    data_train = data_train.reshape(data_train.shape[0],-1).astype(float)
    data_test = data_test.reshape(data_test.shape[0],-1).astype(float)
    
    from sklearn.preprocessing import StandardScaler
    data_train=StandardScaler().fit_transform(data_train)
    data_test=StandardScaler().fit_transform(data_test)  
    return data_train,data_test


def create_datafile(dataset_name,data_train,data_test):
    data_dir = dataset_name
    test_filename = "test.npy"
    if not test_filename in data_dir:
        f = open(os.path.join(data_dir, test_filename), "w")
        choice = np.random.choice(data_test.shape[0], 100, replace=False)
        np.save(os.path.join(data_dir, test_filename),data_test[choice])
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
        else:
            choice = np.random.choice(data_train.shape[0], int(data_train.shape[0]*exp/data_exp), replace=False)
            np.save(filepath,data_train[choice])
        f.close()  

def cifar10(extract_features:bool=False)->None:
    from torchvision.datasets import CIFAR10
    dataset_name = "CIFAR10"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_train = CIFAR10(root=data_dir, train=True, download=True).data.swapaxes(1,3).swapaxes(2,3)
    data_test = CIFAR10(root=data_dir, train=False, download=True).data.swapaxes(1,3).swapaxes(2,3)

    data_train,data_test = dataset2vec(data_train,data_test,extract_features)

    create_datafile(dataset_name,data_train,data_test)      

def MNIST(extract_features:bool=False)->None:
    from torchvision.datasets import MNIST
    dataset_name = "MNIST"
    data_dir = dataset_name
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_train = MNIST(root=data_dir, train=True, download=True).data.numpy()
    data_test = MNIST(root=data_dir, train=False, download=True).data.numpy()

    data_train,data_test = dataset2vec(data_train,data_test,extract_features)

    create_datafile(dataset_name,data_train,data_test)   


if __name__ == "__main__":
    gen_randvec_file(3,4,5,6)
    # cifar10(extract_features=True)
    # MNIST(extract_features=False)