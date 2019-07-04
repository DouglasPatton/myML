import random
import numpy as np

class data_gen():
    '''generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    def __init__(self, xtrain_size=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
        k_1=200
        n=5
        self.xtrain=
        
    
        xtall=xmean
        xwide=xvar
        spreadx=np.random.randint(xwide, size=(1,k_1))+1#random row vector to multiply by each random column of x to allow s.d. upto 5
        shiftx=np.random.randint(0,xtall, size=(1,k_1))-xtall/2#random row vector to add to each column of x
        randx=np.random.randn(n,k_1)
        self.x = np.concatenate((np.ones((n,1)),shiftx+spreadx*randx),axis=1)
        #generate error~N(0,1)
        self.e=np.random.randn(n)*evar**.5
        

        #make beta integer, non-zero
        self.b=(np.random.randint(betamax, size=(k_1+1,))+1)*(2*np.random.randint(2, size=(k_1+1,))-np.ones(k_1+1,)) #if beta is a random integer, it could be 0
        #make a simple y for testing
        self.y=np.matmul(self.x, self.b)+self.e
