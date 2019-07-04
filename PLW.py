import numpy as np
from scipy.optimize import minimize


def myPLWLS(y,x,attr1=x,mse=1,costfn=1):
    print('mse=',mse,'costfn=',costfn)
    if np.allclose(np.ones([attr1.shape[0],]),attr1[:,0]):#fix problems with differencing column of 1's
        attr=attr1[:,1:]
    else: attr=attr1
    n,k=np.shape(x)
    j=np.shape(attr)[1]+1 #plus 1 to go with the column of 1's added after differencing
    attr_tile=np.tile(attr,(n,1,1))# stack the attributes matrix (nxk array) n times
    diffstack=np.transpose(attr_tile,(1,0,2))-attr_tile#transpose the z and x dimensions (listed as z,x,y)
    diffstack=np.concatenate((np.ones([n,n,1]),diffstack), axis=2) # add a column of 1's to all the differences
    wt_try0=np.zeros([j,])
    wt_try0[0]=1
    return minimize(PLWLStryMSE,wt_try0,args=(y,x,diffstack,mse,costfn),method='Nelder-Mead')

def Attr_To_Diffstck(attr):
    n,j=np.shape(attr); j+=1 #plus 1 to go with the column of 1's added after differencing
    attr_tile=np.tile(attr,(n,1,1))# stack the attributes matrix (nxk array) n times
    diffstack=np.transpose(attr_tile,(1,0,2))-attr_tile#transpose the z and x dimensions (listed as z,x,y)
    return np.concatenate((np.ones([n,n,1]),diffstack), axis=2)


def Reg_wts_stack(corrparams,diffstack):
    #return np.sum(((2**corrparams*diffstack)**2),axis=2)**-0.5)
    #weighted, squared, summed across wt parameter, j, rooted and flipped
    regwt_stack=np.sum(np.exp((corrparams*diffstack)**2),axis=2)**(-0.5)
    return regwt_stack/(np.sum(regwt_stack**2)**.5/regwt_stack.shape[0])#normalize the weights seems to help convergence
    
def PLWLScastMSE(corrparams,attr1,y,x):
    #use corrparams, correspondence attributes, known y and x to calculate mse for the estimator/corrparams
    if np.allclose(np.ones([attr1.shape[0],]),attr1[:,0]):#fix problems with differencing column of 1's
        attr=attr1[:,1:]
    else: attr=attr1
    n,k=np.shape(x)
    regwt_stack=Reg_wts_stack(corrparams,Attr_To_Diffstck(attr))
    
    x_wt_stck=np.tile(regwt_stack.reshape(n,n,1),(1,1,k))*np.tile(x,(n,1,1))
    y_wt_stck=regwt_stack.reshape(n,n,1)*np.tile(y.reshape(n,1),(n,1,1))
    x_wt_stckT=np.transpose(x_wt_stck,(0,2,1))#transpose X aross the stack for x'x
    Qi=np.linalg.inv(x_wt_stckT@x_wt_stck)
    Bstack=Qi@x_wt_stckT@y_wt_stck
    #print('first 5 params=',Bstack[0:5,:,:])
    
                             
    ystack=np.tile(y.reshape(n,1),(n,1,1))
    xstack=np.tile(x,(n,1,1))
    err=(ystack-xstack@Bstack)#errors are not calculated using weighted y and x
    
    yhat_stack=xstack@Bstack                         
    yhats=yhat_stack.diagonal().diagonal()
    err_diag=err.diagonal().diagonal()
    mse=np.sum(err_diag**2)/err_diag.size
    '''p=figure()
    p.circle(x[:,1],y)
    p.square(x[:,1],yhats)
    p.show()'''
    #print('mse_cast=',mse)
    return mse
    

    
def PLWLStryMSE(corrparams,y,x,diffstack,MSE=1,costfn=1):
    #print(x.shape)
    print('corrparams=',corrparams)
    n,k=np.shape(x)
    j=np.shape(corrparams)[0]
    #print(k,j)
    regwt_stack=Reg_wts_stack(corrparams,diffstack)
    x_wt_stck=np.tile(regwt_stack.reshape(n,n,1),(1,1,k))*np.tile(x,(n,1,1))#the stack of weights is broadcast and multiplied by stack of x.(z,x,y) indexing
    y_wt_stck=regwt_stack.reshape(n,n,1)*np.tile(y.reshape(n,1),(n,1,1))#the stack of weights is broadcast and multiplied by stack of y.
    x_wt_stckT=np.transpose(x_wt_stck,(0,2,1))#transpose X aross the stack
    Qi=np.linalg.inv(x_wt_stckT@x_wt_stck)
    #print(y_wt_stck.shape)
    Bstack=Qi@x_wt_stckT@y_wt_stck #use weighted data to calculate gls parameters
    #print('first params=',Bstack[0,:,:])
    ystack=np.tile(y.reshape(n,1),(n,1,1))
    xstack=np.tile(x,(n,1,1))
    err=(ystack-xstack@Bstack)#errors are not calculated using weighted y and x
    if costfn==1: cost_err=err**2-regwt_stack**(-2) #squared error prediction
    if costfn==2: cost_err=err
    corr_err=err**2-regwt_stack**(-2)
    if MSE==1:
        out=np.sum(cost_err**2)/cost_err.size
        print('{}MSE all='.format(costfn),out)
        return out
    if MSE==2:
        wt_cost_err=(cost_err)*regwt_stack
        out=np.sum(wt_cost_err**2)/wt_cost_err.size
        print('{}MSE wt_all='.format(costfn),out)
        return out
    if MSE==3:
        diag_err=cost_err.diagonal().diagonal()
        print('diag',diag_err.shape)
        out=np.sum(diag_err**2)/diag_err.size
        print('{}MSE n_only='.format(costfn),out)
        return out
    
def PLWLSpredict(corrparams,attr1,y,x,attr1_val,y_val,x_val):
    #use corrparams, correspondence attributes, known y and x to calculate mse for the estimator/corrparams
    if np.allclose(np.ones([attr1.shape[0],]),attr1[:,0]):#fix problems with differencing column of 1's
        attr=attr1[:,1:]
    else: attr=attr1
    if np.allclose(np.ones([attr1_val.shape[0],]),attr1_val[:,0]):#fix problems with differencing column of 1's
        attr_val=attr1_val[:,1:]    
    else: attr_val=attr1_val
    n,k=np.shape(x)
    nv,k1=np.shape(x_val)
    j=np.shape(attr)[1]+1 #plus 1 to go with the column of 1's added after differencing
    ctr_attr_stck=np.tile(attr_val,(n,1,1))# stack the centered attributes matrix (nxk array) n times
    diffstack0=np.transpose(ctr_attr_stck,(1,0,2))-np.tile(attr,(nv,1,1))#transpose the z and x dimensions (listed as z,x,y)
    diffstack=np.concatenate((np.ones([nv,n,1]),diffstack0), axis=2) ##so diffstack is shape nv,n,j
    
    regwt_stack=Reg_wts_stack(corrparams,diffstack)
    x_wt_stck=np.tile(regwt_stack.reshape(nv,n,1),(1,1,k))*np.tile(x,(nv,1,1))
    y_wt_stck=regwt_stack.reshape(nv,n,1)*np.tile(y.reshape(n,1),(nv,1,1))
    x_wt_stckT=np.transpose(x_wt_stck,(0,2,1))#transpose X aross the stack for x'x
    Qi=np.linalg.inv(x_wt_stckT@x_wt_stck)
    Bstack=Qi@x_wt_stckT@y_wt_stck
    #print('first 5 params=',Bstack[0:5,:,:])
                                 
    ystack=np.tile(y_val.reshape(nv,1),(nv,1,1))
    xstack=np.tile(x_val,(nv,1,1))
    err=(ystack-xstack@Bstack)#errors are not calculated using weighted y and x
    
    yhat_stack=xstack@Bstack                         
    yhats=yhat_stack.diagonal().diagonal()
    err_diag=err.diagonal().diagonal()
    mse=np.sum(err_diag**2)/err_diag.size
    '''p=figure()
    p.circle(x[:,1],y)
    p.square(x[:,1],yhats)
    p.show()'''
    #print('mse_cast=',mse)
    return mse    
