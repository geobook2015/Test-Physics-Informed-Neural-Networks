
import sys 
sys.path.insert(0, 'Utilities/')

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
    #weights = np.reshape(tmp[0:q2+q], (q+1,q)) 
    times = dt * tmp[q2+q:] 
    times = np.array([0.] + times[:, 0].tolist())[:, None] 
    return times

def learning(m, epoch, dt, q, cond_b, cond_i, param):

    model = m(net, dt, cond_b, cond_i, param, q)
    model.train(epoch)

    return model
def main1_2d(x, y, model):

    x_star, y_star = np.meshgrid(x, y)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]

    umap_pred = model.predict(x_star, y_star)
    umap_pred = np.reshape(umap_pred, (x.shape[0], y.shape[0], q+1))
    #umap_pred = umap_pred[:, :, 0:-1]
    print('x_star', x_star.shape)
    print('umap_pred', umap_pred.shape)

    #umap_pred = np.flip(umap_pred, 0)
    return umap_pred
def main2(xy, t, func): 
    x = xy[0] 
    y = xy[1]

    dt = t[1, 0] - t[0, 0]
    dx = x[1, 0] - x[0, 0]
    X, Y, T = np.meshgrid(x, y, t)
    T = T.flatten()[:, None]
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]

    U = func(X, Y, T)
    print(U.shape)
    umap = np.reshape(U, (x.shape[0], y.shape[0], t.shape[0]))
    #umap = np.reshape(U, (t.shape[0], x.shape[0], y.shape[0]))
    return umap

def func1(c, D): 
    f = lambda x, y, t : np.exp(- 2 * t) * np.sin(x) * np.sin(y) 
    return f

def func2(c, D): 
    m = 3 
    n = 4 
    l2 = D * ((m * np.pi)**2 + (n * np.pi)**2) 
    f = lambda x, y, t : np.exp(-l2 * t) * np.sin(m * np.pi * x) * np.sin(n * np.pi * y) 
    return f

def func3(c, D):

    def f(x, y, t):
        ret = 0
        m_list = [9, 4]
        n_list = [3, 1]
        A = np.array([300, 500, 700, 200]) * 1e12
        i = 0
        for m in m_list:
            for n in n_list:
                l2 = D * ((m * np.pi)**2 + (n * np.pi)**2)
                ret += A[i] * np.sin(m * np.pi, x) * np.sin(n * np.pi * y) * np.exp(-l2 * t)
                i += 1
            
        return ret

    return f
def translate_func_2d(func): 
    f = lambda ret : func(ret[:, 0], ret[:, 1], 0) 
    return f

if __name__ == "__main__":

    epoch = 1000
    #epoch = 4000
    #epoch = 40000
    #epoch = 100000

    #if True:
    if False:
        c = 0.0
        D = 1.0
        func = func1(c, D)
        func_init = translate_func_2d(func)

        N0 = 200
        Nb = 200
        #wiki
    
        line1 = sampling.Line_AxisX_2d(0, 0, np.pi)
        line2 = sampling.Line_AxisY_2d(0, 0, np.pi)
        line3 = sampling.Line_AxisX_2d(np.pi, 0, np.pi)
        line4 = sampling.Line_AxisY_2d(np.pi, 0, np.pi)
        cond_b = sampling.dirichlet([line1, line2, line3, line4], 0, Nb)
    
        space_test = sampling.Rectangle([0, 0], [1, 1], 0.0)
        space = sampling.Rectangle([0, 0], [np.pi, np.pi], 1e-6)
        cond_i = sampling.volume([space], func_init, N0)


        dt = 3e-3
        x_step = 4e-3
    
        q = 500
        layers = [2, 50, 50, 50, 50, 50, 50, q+1]
        net = NeuralNet.Linear(layers)
        model = pde.PINN_Diffusion
        main1 = main1_2d
    

    elif True:
    #elif False:
        c = 0.0
        D = 8.0
        func = func2(c, D)
        func_init = translate_func_2d(func)

        N0 = 200
        Nb = 200
    
        #wiki
        upper = 1.0
        line1 = sampling.Line_AxisX_2d(0, 0, upper)
        line2 = sampling.Line_AxisY_2d(0, 0, upper)
        line3 = sampling.Line_AxisX_2d(upper, 0, upper)
        line4 = sampling.Line_AxisY_2d(upper, 0, upper)
        #cond_b = sampling.dirichlet([line1, line2, line3, line4], 0, Nb)
        cond_b = sampling.dirichlet([line1, line2, line3, line4], func_init, Nb)
        #cond_b = sampling.Nothing()
    

        space = sampling.Rectangle([0, 0], [1, 1], 1e-6)
        space_test = sampling.Rectangle([0, 0], [1, 1], 0.0)
        cond_i = sampling.volume([space], func_init, N0)


        dt = 1e-3
        x_step = 4e-3
        q = 500
        #layers = [2, 200, 200, 200, 200, q+1]
        layers = [2, 50, 50, 50, 50, 50, 50, q+1]
    
    
        net = NeuralNet.Linear(layers)
        model = pde.PINN_Diffusion
        main1 = main1_2d


    elif False:
    #elif True:
    
        c = 0.0
        D = 2.0
        func = func3(c, D)
        func_init = translate_func_2d(func)

        N0 = 200
        Nb = 200
        #wiki
    
        line1 = sampling.Line_AxisX_2d(0, 0, 1)
        line2 = sampling.Line_AxisY_2d(0, 0, 1)
        line3 = sampling.Line_AxisX_2d(1, 0, 1)
        line4 = sampling.Line_AxisY_2d(1, 0, 1)
        cond_b = sampling.dirichlet([line1, line2, line3, line4], 0, Nb)
    
        #space = sampling.Rectangle([0, 0], [1, 1], 1e-6)
        space = sampling.Rectangle([0, 0], [1, 1], 0.0)
        cond_i = sampling.volume([space], func_init, N0)

        dt = 1e-3
        x_step = 4e-3
    
        q = 500
        #layers = [2, 200, 200, 200, 200, q+1]
        layers = [2, 50, 50, 50, 50, 50, 50, q+1]
    
        net = models_2d.NeuralNet(layers)
        model = models_2d.PINN_Diffusion
        main1 = main1_2d


    N_test = 200
    xy = space_test.sampling_diagonal(N_test)
    x_test = xy[:, 0][:, None]
    y_test = xy[:, 1][:, None]
    print(y_test)

    q = 500
    t = get_times(q, dt)

    print('time', t.shape)

    y_true = main2([x_test, y_test], t, func)
    y_pred = np.copy(y_true)

    model = learning(model, epoch, dt, q, cond_b, cond_i, [c, D])
    y_pred = main1(x_test, y_test, model)


    print(y_true.shape)

    y_abs_max = np.max(np.abs(y_true))
    #y_true_max = np.max(y_true) - 0.2
    y_true_max = np.max(y_true) - 0.0
    y_true_min = np.min(y_true)

    y_ErrMax = np.max(np.abs(y_true - y_pred))+0.001
    print('y_ErrMax', y_ErrMax)

    pl.figure()
    for loop in range(0, 500, 50):

        print('t', t[loop])
        #y_true = main2([x_test, y_test], t[loop], func)
        #y_pred = np.copy(y_true)
    
        pl.clf()

        pl.subplot(131)
        pl.imshow(y_true[:, :, loop])
        pl.colorbar()
        pl.clim([y_true_min, y_true_max])


        pl.subplot(132)
        pl.imshow(y_pred[:, :, loop])
        pl.colorbar()
        pl.clim([y_true_min, y_true_max])

        pl.subplot(133)
        pl.imshow(np.abs(y_true[:, :, loop] - y_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, y_ErrMax])
        pl.show()

