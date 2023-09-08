import numpy as np
import os,time
def gen_randvec(dim:int)->np.array:
    return np.random.randint(-2048,2048,size=dim)

def gen_randvec_file(*exp):
    if not os.path.exists("randvec"):
        os.makedirs("randvec")
    dims = [128]
    for dim in dims:
        for i in exp:
            dir_path = os.path.join("randvec",f"10^{i}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            filename = f"randvec{dim}_10^{i}.csv"
            if filename in dir_path:
                continue
            filepath : str = os.path.join(dir_path, filename)
            f = open(filepath, "w")
            scale = 10**i
            t = time.time()
            for j in range(int(scale)):
                vec = gen_randvec(dim)
                line : str = np.array2string(vec,separator=',')
                line = line[1:-1].replace(' ',"").replace("\n","")+'\n'
                f.write(line)
            t = time.time() - t
            print(f"File {filename} generated in {t} seconds.")
            f.close()
if __name__ == "__main__":
    gen_randvec_file(3,4,5)