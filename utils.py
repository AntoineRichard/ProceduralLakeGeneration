import scipy.stats as st
import numpy as np
import pickle

def loadDict(path):    
    with open(path, 'rb') as handle:
        parser = pickle.load(handle)
    return parser

def saveDict(dct, name):
    with open(name+'.pkl', 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def makeGaussianKernel(width, nsig):
    x = np.linspace(-nsig, nsig, width+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.max()
    
def saveListToText(name, data):
    with open(name,'w') as f:
        f.write('\n'.join(data))

def randomSign():
    return np.sign(np.random.rand() - 0.5)