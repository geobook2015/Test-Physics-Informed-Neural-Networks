
import sys 
sys.path.insert(0, '../Utilities/')

import tensorflow as tf 
import numpy as np
import time 
import scipy.io 
import sampling 
#import matplotlib.pyplot as plt 
#from plotting import newfig, savefig 
#import matplotlib.gridspec as gridspec 
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl 
import pde 
import NeuralNet

np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

def get_times(q, dt): 
    tmp = np.float32(np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))     
    #weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    times = dt * tmp[q**2+q:]
    times = np.array([0.] + times[:, 0].tolist())[:, None]
    return times

def make_model(m, net, dt, q, cond_b, cond_i, param):

    model = m(net, dt, cond_b, cond_i, param, q)
    return model

def main1_2d(x, y, model):

    x_star, y_star = np.meshgrid(x, y)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]

    u_pred, v_pred = model.predict(x_star, y_star)
    u_pred = np.reshape(u_pred, (x.shape[0], y.shape[0], q+1))
    v_pred = np.reshape(v_pred, (x.shape[0], y.shape[0], q+1))
    print('x_star', x_star.shape)
    print('u_pred', u_pred.shape)
    print('v_pred', v_pred.shape)

    #umap_pred = np.flip(umap_pred, 0)
    return u_pred, v_pred


def main2(xy, t, func): 
    x = xy[0]
    y = xy[1]

    dt = t[1, 0] - t[0, 0]
    dx = x[1, 0] - x[0, 0]
    X, Y, T = np.meshgrid(x, y, t)
    T = T.flatten()[:, None]
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]

    U, V = func(X, Y, T)
    #
    umap = np.reshape(U, (x.shape[0], y.shape[0], t.shape[0]))
    vmap = np.reshape(V, (x.shape[0], y.shape[0], t.shape[0]))
    return umap, vmap

def func_burgers2(Re):

    def f(x, y, t):
        temp = (-4 * x + 4 * y - t) * Re / 32.
        temp2 = 4 * (1 + np.exp(temp))
        U = 3. / 4. -  1. / temp2
        V = 3. / 4. +  1. / temp2
        return [U, V]

    return f


def translate_func_2d(func):
    f = lambda ret : func(ret[:, 0], ret[:, 1], 0) 
    return f


if __name__ == "__main__":

    print("start!!")
    epoch = 200
    #epoch = 4000
    #epoch = 40000
    #epoch = 100000
    q = 500



    if True:
    #if False:
        #Re = 0.001
        #Re = 20.0
        #Re = 50.0
        Re = 4.0
        #Re = 1e-5
        #Re = 1e+5
        func = func_burgers2(Re)
        uv_func = translate_func_2d(func)
    
        N0 = 200
        Nb = 200
    
        upper = 0.5
        line1 = sampling.Line_AxisX_2d(0, 0, upper)
        line2 = sampling.Line_AxisY_2d(0, 0, upper)
        line3 = sampling.Line_AxisX_2d(upper, 0, upper)
        line4 = sampling.Line_AxisY_2d(upper, 0, upper)
        cond_b = sampling.dirichlet([line1, line2, line3, line4], uv_func, Nb)

        space = sampling.Rectangle([0, 0], [upper, upper], 1e-8)
        space_test = sampling.Rectangle([0, 0], [upper, upper], 0.0)
        cond_i = sampling.dirichlet([space], uv_func, N0)

        #dt = 0.3
        #dt = 0.5
        #dt = 0.3
        #dt = 1.0
        dt = 2.0
        #dt = 0.6
        #dt = 1e-2
        x_step = 4e-3
    
        q = 500
        layers = [2, 50, 50, 50, 50, 50, 50, (q+1)*2]
        #layers = [2, 200, 200, 200, 200, (q+1)*2]
        net = NeuralNet.Linear(layers)
        main1 = main1_2d
        model = pde.PINN_Burgers
        param = [Re]
        #


    #elif True:
    elif False:
        Re = 0.001
        func = func_burgers2(Re)
        uv_func = translate_func_2d(func)
    
        N0 = 200
        Nb = 100
    
        upper = 0.5
        line1 = sampling.Line_AxisX_2d(0, 0, upper)
        line2 = sampling.Line_AxisY_2d(0, 0, upper)
        line3 = sampling.Line_AxisX_2d(upper, 0, upper)
        line4 = sampling.Line_AxisY_2d(upper, 0, upper)
        cond_b = sampling.dirichlet([line1, line2, line3, line4], uv_func, Nb)

        space = sampling.Rectangle([0, 0], [upper, upper], 1e-8)
        space_test = sampling.Rectangle([0, 0], [upper, upper], 0.0)
        cond_i = sampling.dirichlet([space], uv_func, N0)

        dt = 2e+0
        x_step = 4e-3
    
        q = 500
        #layers = [2, 50, 50, 50, 50, 50, 50, q+1]
        layers = [2, 200, 200, 200, 200, q+1]
        net1 = NeuralNet.Linear(layers)
        net2 = NeuralNet.Linear(layers)
        net = [net1, net2]
        main1 = main1_2d
    
        param = [Re]


    #elif True:
    elif False:
        #Burgers 2D
        #https://arxiv.org/ftp/arxiv/papers/1409/1409.8673.pdf
        #https://www.sciencedirect.com/science/article/pii/S0898122110003883
        
        #Re = 100.0
        Re = 0.001
        func = func_burgers2(Re)
        u_init = lambda ret : (np.sin(np.pi * ret[:, 0]) + np.cos(np.pi * ret[:, 1]))
        v_init = lambda ret : (ret[:, 0] + ret[:, 1])
        
        N0 = 200
        Nb = 200
        
        #wiki
        
        upper = 0.5
        line1 = sampling.Line_AxisX_2d(0, 0, upper)
        line2 = sampling.Line_AxisY_2d(0, 0, upper)
        line3 = sampling.Line_AxisX_2d(upper, 0, upper)
        line4 = sampling.Line_AxisY_2d(upper, 0, upper)
        b_u = []
        b_u.append(lambda ret : np.cos(np.pi * ret[:, 1][:, None]))
        b_u.append(lambda ret : 1 + np.sin(np.pi * ret[:, 0][:, None]))
        b_u.append(lambda ret : 1 + np.cos(np.pi * ret[:, 1][:, None]))
        b_u.append(lambda ret : np.sin(np.pi * ret[:, 0][:, None]))
        b_v = []
        b_v.append(lambda ret : ret[:, 1][:, None])
        b_v.append(lambda ret : ret[:, 0][:, None])
        b_v.append(lambda ret : 0.5 + ret[:, 1][:, None])
        b_v.append(lambda ret : 0.5 + ret[:, 0][:, None])
        cond_b = sampling.dirichlet([line1, line2, line3, line4], [b_u, b_v], Nb)
    
        space = sampling.Rectangle([0, 0], [upper, upper], 1e-8)
        space_test = sampling.Rectangle([0, 0], [upper, upper], 0.0)
        cond_i = sampling.dirichlet([space], [u_init, v_init], N0)
    
        dt = 3e+0
        x_step = 4e-3
        
        q = 500
        #layers = [2, 50, 50, 50, 50, 50, 50, q+1]
        layers = [2, 200, 200, 200, 200, q+1]
        net1 = NeuralNet.Linear(layers)
        net2 = NeuralNet.Linear(layers)
        net = [net1, net2]
        main1 = main1_2d
        param = [Re]
    


    t = get_times(q, dt)

    N_test = 200
    xy = space_test.sampling_diagonal(N_test)
    x_test = xy[:, 0][:, None]
    y_test = xy[:, 1][:, None]
    print('x_test, y_test', x_test.shape, y_test.shape)


    print('time', t.shape)

    U_true, V_true = main2([x_test, y_test], t, func)
    U_pred, V_pred = np.copy(U_true), np.copy(V_true)

    model = make_model(pde.PINN_Burgers, net, dt, q, cond_b, cond_i, param)
    model.train(10)
    #model.load()
    U_pred, V_pred = main1(x_test, y_test, model)


    print('U_true.shape', U_true.shape)

    u_abs_max = np.max(np.abs(U_true))
    u_true_max = np.max(U_true) - 0.0
    u_true_min = np.min(U_true)

    v_abs_max = np.max(np.abs(V_true))
    v_true_max = np.max(V_true) - 0.0
    v_true_min = np.min(V_true)


    U_ErrMax = np.max(np.abs(U_true - U_pred))+0.001
    V_ErrMax = np.max(np.abs(V_true - V_pred))+0.001
    print('U_ErrMax, V_ErrMax', U_ErrMax, V_ErrMax)

    pl.figure()
    for loop in range(0, 500, 50):

        print('t', t[loop])
        pl.clf()

        pl.subplot(231)
        pl.imshow(U_true[:, :, loop])
        pl.colorbar()
        pl.clim([u_true_min, u_true_max])
    
        pl.subplot(232)
        pl.imshow(U_pred[:, :, loop])
        pl.colorbar()
        pl.clim([u_true_min, u_true_max])

        pl.subplot(233)
        pl.imshow(np.abs(U_true[:, :, loop] - U_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, U_ErrMax])
   

        pl.subplot(234)
        pl.imshow(V_true[:, :, loop])
        pl.colorbar()
        pl.clim([v_true_min, v_true_max])

        pl.subplot(235)
        pl.imshow(V_pred[:, :, loop])
        pl.colorbar()
        pl.clim([v_true_min, v_true_max])

        pl.subplot(236)
        pl.imshow(np.abs(V_true[:, :, loop] - V_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, V_ErrMax])
        
    
    
        pl.show()
