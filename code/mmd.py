import torch
import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import RBF

def mmd(X,Y,sigma):

    shapeX = X.size()
    dims_to_reduceX = list(shapeX)[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimX = reduce(lambda x,y: x*y, dims_to_reduceX)

    shapeY = Y.size()
    dims_to_reduceY = list(shapeY)[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimY = reduce(lambda x,y: x*y, dims_to_reduceY)
   
    X = X.view(shapeX[0], reduced_dimX)
    print(X.size())
    Y = Y.view(shapeY[0], reduced_dimY)

    K_XX = rbf_kernel(X,X,sigma)
    K_XY = rbf_kernel(X,Y,sigma)
    K_YY = rbf_kernel(Y,Y,sigma)

    return np.mean(K_XX) -2 * np.mean(K_XY) + np.mean(K_YY)


def test_mmd():

    a = torch.tensor([[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]])
    b = torch.tensor([[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]])

    c = torch.tensor([[.5,.1,.1,.1,.2],[.3,.2,.1,.0,.4]])
    d = torch.tensor([[.2,.1,.1,.4,.2],[.3,.2,.1,.0,.4]])

    c_2 = torch.tensor([[5.,1.,0.3,5.2,0.1,1.,5.2,5.1,5.1,0.5]])
    d_2 = torch.tensor([[1.,0.7,1.2,1.9,1.2,0.95,0.1,0.2,0.1,0.4]])

    e = torch.tensor([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]], [[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f = torch.tensor([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]], [[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])

    c_1 = torch.tensor([[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.]])
    d_1 = torch.tensor([[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.]])

    e_1 = torch.tensor([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]]], [[[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f_1 = torch.tensor([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]]], [[[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])

    aa = e.view(1,30)
    bb = f.view(1,30)

    # print(f"aa: {aa}")
    # print(f"bb: {bb}")

    #t_1 = mmd(e,f)
    #t_2 = mmd2(e_1,f_1)
    #t_3 = mmd(a,b)
    #t_4 = mmd2(a,b)
    #t_5 = mmd(c,d)
    #t_6 = mmd2(c,d)

    #t_7 = mmd2(c_1,d_1)

    #print(f'mmd(e,f): {t_1}')
    #print(f'mmd2(e_1,f_1): {t_2}')
    #print(f'mmd(a,b): {t_3}')
    #print(f'mmd2(a,b): {t_4}')
    #print(f'mmd(c,d): {t_5}')
    #print(f'mmd2(c,d): {t_6}')
    #print(f'mmd2(c_1,d_1): {t_7}')

    '''
    kl = tf.keras.losses.KLDivergence()
    m = (1/2) * (c + d)
    print(f'kl result_cd: {kl(c,m).numpy}')
    print(f'kl result_dc: {kl(d,m).numpy}')
    print(f'kl result_cdm: {0.5 * kl(c,m) + 0.5 * kl(d,m)}')
    '''

    kernel = 1.0 * RBF(1.0)
    res_of_RBF = kernel(c_2, d_2)
    # print(f"res_of_RBF: {res_of_RBF}")

    # print(f"rbf_kernel(c_2,d_2,0.5): {rbf_kernel(c_2,d_2,0.5)}")

    # print(f"rbf_kernel(c_2,c_2,0.5): {rbf_kernel(c_2,c_2,0.5)}")

    res = mmd(a,b,2.)
    print(res)

test_mmd()
    

