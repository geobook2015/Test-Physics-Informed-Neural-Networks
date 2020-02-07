
import sys
sys.path.insert(0, 'Utilities/')
import tensorflow as tf 
import numpy as np 
import scipy.io 
import sampling 
import time

class PINN(object):

# Initialize the class
    def __init__(self, neural_net, dt, q):

        self.dt = dt
        self.q = max(q, 1)



        # Initialize NN
        #self.NN = self.NeuralNet(layers)
        self.NN = neural_net
    
        # Load IRK weights
        tmp = np.float32(np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        self.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        self.IRK_times = tmp[q**2+q:]
    
    
        #self.renew_random = None
        #self.renew_tf = None
        self.sess = None
        self.train_op_Adam = None
        self.tf_dict = None
        self.optimizer = None

    def renew_random(self):
        return NotImplementedError()
    
    def renew_tf(self):
        return NotImplementedError()
    
    def load(self):
        #init = tf.global_variables_initializer()
        #self.sess.
        print(self.path + '.meta')
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            saver.restore(self.sess, self.path + '.meta')
    
    
    
    def train(self, nIter):

        start_time = time.time()
        saver = tf.train.Saver()


        #self.renew_ordered()


        interval = 1
        for it in range(nIter):
            if it % interval == 0:
                self.renew_random()
                pass

            self.renew_tf()

            #if it % 50000 == 0:
            if False:
                self.optimizer_Adam._lr *= 0.1
            self.sess.run(self.train_op_Adam, self.tf_dict)
        
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, self.tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

        if True:
            self.renew_random()
            self.renew_tf()
            self.optimizer.minimize(self.sess,
                                    feed_dict = self.tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)

        save_path = saver.save(self.sess, self.path)
        #self.sess.close()

class PINN_Diffusion(PINN): # Initialize the class 
    def __init__(self, neural_net, dt, cond_b, cond_0, param_pde, q):

        super(PINN_Diffusion, self).__init__(neural_net, dt, q)
    
        #self.name = 'disffusion'
        self.path = "model//diff2d.ckpt"
    
        self.cond_0 = cond_0
        self.cond_b = cond_b
        self.bound_l = self.cond_0.lower[0]
        self.bound_u = self.cond_0.upper[0]
    
        self.c = param_pde[0]
        self.D = param_pde[1]
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                 log_device_placement=True))
    
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.xb_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.yb_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.u0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.ub_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_y0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_xb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))
        self.dummy_yb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))


        self.U0_pde = self.net_U0(self.x0_tf, self.y0_tf) # N x (q+1)
        self.Ub_pred = self.net_U1(self.xb_tf, self.yb_tf) # 2 x (q+1)
    
        # PDE Condition + initial condition + Boundary Condition
        self.loss = tf.reduce_sum(tf.square(self.U0_pde - self.u0_tf))
        if self.cond_b.N > 0:
            self.loss += tf.reduce_sum(tf.square(self.Ub_pred - self.ub_tf))
            #self.loss += tf.reduce_sum(tf.square(self.Ub_pde - self.ub_tf))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        lr = 1e-4
        self.optimizer_Adam = tf.train.AdamOptimizer(lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    
        init = tf.global_variables_initializer()
        self.sess.run(init)
    

    def net_U0(self, x, y):

        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
        U_net = self.NN.forward(X)# N x (q+1)
        U = U_net[:, :-1]# (N, q+1) -> (N, q)
        U_x = self.fwd_gradients_x0(U, x)
        U_y = self.fwd_gradients_y0(U, y)
        U_xx = self.fwd_gradients_x0(U_x, x)
        U_yy = self.fwd_gradients_y0(U_y, y)

        c = self.c
        D = self.D
        #F = - c * (U_x + U_y) + D * (U_xx + U_yy)
        F = D * (U_xx + U_yy)

        #prediction of U0 using Runge-Kutta
        U0_pde = U_net - self.dt * tf.matmul(F, self.IRK_weights.T)# (N, q) * (q+1, q)T -> (N, q+1)
        return U0_pde


    def net_U1(self, x, y):
        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
        U1 = self.NN.forward(X)
        return U1 # N x (q+1)

    def preprocess(self, X):
        H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0
        return H
    
    def callback(self, loss):

        self.renew_random()
        self.renew_tf()
        print('Loss:', loss)


    def renew_random(self):

        sp_b, ub = self.cond_b.generate()
        self.ub = ub
        self.xb = sp_b[:, 0][:, None]
        self.yb = sp_b[:, 1][:, None]

        sp_0, u0 = self.cond_0.generate()
        self.u0 = u0
        self.x0 = sp_0[:, 0][:, None]
        self.y0 = sp_0[:, 1][:, None]


    def renew_tf(self):
        self.tf_dict = {self.x0_tf: self.x0, 
                        self.y0_tf: self.y0, 
                        self.u0_tf: self.u0,
                        self.xb_tf: self.xb,
                        self.yb_tf: self.yb,
                        self.ub_tf: self.ub,
                        self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_y0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_xb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        self.dummy_yb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        }


    def predict(self, x_star, y_star):
    
        U1_star = self.sess.run(self.Ub_pred, {self.xb_tf: x_star, self.yb_tf: y_star})
                
        return U1_star

    def save(self, path):
        pass

    def load(self):
        from tensorflow.saved_model import tag_constants
        with tf.Session(graph=tf.Graph()) as sess:
            #tf.saved_model.loader.load(self.sess, [tag_constants.TRAINING], "model//diff2_" + self.name + ".ckpt")
            tf.saved_model.loader.load(self.sess, "model//diff2_" + self.name + ".ckpt")


    def fwd_gradients_x0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]

    def fwd_gradients_xb(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_xb_tf)[0]
        return tf.gradients(g, self.dummy_xb_tf)[0]
        
    def fwd_gradients_y0(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_y0_tf)[0]
        return tf.gradients(g, self.dummy_y0_tf)[0]

    def fwd_gradients_yb(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_yb_tf)[0]
        return tf.gradients(g, self.dummy_yb_tf)[0]


class PINN_Burgers(PINN): # Initialize the class 
    def __init__(self, neural_net, dt, cond_b, cond_0, param_pde, q): 
        super(PINN_Burgers, self).__init__(neural_net, dt, q)

        self.path = "model//burgers2d.ckpt"
        self.Re = param_pde[0]
        #self.NN = neural_net[0]
    
        self.cond_b = cond_b
        self.cond_i = cond_0
    
        # tf placeholders and graph
        #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                                             log_device_placement=True))
    
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.xb_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.yb_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_y0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_xb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))
        self.dummy_yb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))

        self.u0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.ub_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.vb_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U0_pde, self.V0_pde = self.net_U0(self.x0_tf, self.y0_tf) # N x (q+1)
        self.Ub_pred , self.Vb_pred = self.net_U1(self.xb_tf, self.yb_tf) # 2 x (q+1)


        # PDE Condition + initial condition + Boundary Condition
        self.loss = tf.reduce_sum(tf.square(self.U0_pde - self.u0_tf))
        self.loss += tf.reduce_sum(tf.square(self.V0_pde - self.v0_tf))
        self.loss += tf.reduce_sum(tf.square(self.Ub_pred - self.ub_tf))
        self.loss += tf.reduce_sum(tf.square(self.Vb_pred - self.vb_tf))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,#50000
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        #lr = 1e-4
        #lr = 1e-3
        lr = 4e-3
        self.optimizer_Adam = tf.train.AdamOptimizer(lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.sess.run(init)
    
    

    def net_U0(self, x, y):
        q = self.q
        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
    
        U_V = self.NN.forward(X)
        U_net = U_V[:, :q+1]
        V_net = U_V[:, (q+1):(2*(q+1))]
    
        U = U_net[:, :-1]# (N, q+1) -> (N, q)
        U_x = self.fwd_gradients_x0(U, x)
        U_y = self.fwd_gradients_y0(U, y)
        U_xx = self.fwd_gradients_x0(U_x, x)
        U_yy = self.fwd_gradients_y0(U_y, y)
    
        V = V_net[:, :-1]# (N, q+1) -> (N, q)
        V_x = self.fwd_gradients_x0(V, x)
        V_y = self.fwd_gradients_y0(V, y)
        V_xx = self.fwd_gradients_x0(V_x, x)
        V_yy = self.fwd_gradients_y0(V_y, y)

        F_U = - (U * U_x + V * U_y) + (U_xx + U_yy) / self.Re
        F_V = - (U * V_x + V * V_y) + (V_xx + V_yy) / self.Re

        #prediction of U0 using Runge-Kutta
        U0_pde = U_net - self.dt * tf.matmul(F_U, self.IRK_weights.T)# (N, q) * (q+1, q)T -> (N, q+1)
        V0_pde = V_net - self.dt * tf.matmul(F_V, self.IRK_weights.T)# (N, q) * (q+1, q)T -> (N, q+1)
        return U0_pde, V0_pde


    def net_U1(self, x, y):
        q = self.q
        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
    
        U_V = self.NN.forward(X)
        U1 = U_V[:, :q+1]
        V1 = U_V[:, (q+1):(2*(q+1))]
    
        return U1, V1 # N x (q+1)


    def predict(self, x_star, y_star):
    
        U = self.sess.run(self.Ub_pred, {self.xb_tf: x_star, self.yb_tf: y_star})
        V = self.sess.run(self.Vb_pred, {self.xb_tf: x_star, self.yb_tf: y_star})
                
        return U, V
    
    
    def callback(self, loss):

        self.renew_random()
        self.renew_tf()
        print('Loss:', loss)

    def renew_random(self):

        sp_b, bound = self.cond_b.generate()
        #print(bound[0].shape)
        self.ub = bound[0][:, None]
        self.vb = bound[1][:, None]
        self.xb = sp_b[:, 0][:, None]
        self.yb = sp_b[:, 1][:, None]

        sp_0, init = self.cond_i.generate()
        self.u0 = init[0][:, None]
        self.v0 = init[1][:, None]
        self.x0 = sp_0[:, 0][:, None]
        self.y0 = sp_0[:, 1][:, None]
    
        #print(self.xb.shape, self.yb.shape)
        #print(self.ub.shape, self.vb.shape)
        #print(self.x0.shape, self.y0.shape)
        #print(self.u0.shape, self.v0.shape)


    def renew_tf(self):
        self.tf_dict = {self.x0_tf: self.x0, 
                        self.y0_tf: self.y0, 
                        self.u0_tf: self.u0,
                        self.v0_tf: self.v0,
                        self.xb_tf: self.xb,
                        self.yb_tf: self.yb,
                        self.ub_tf: self.ub,
                        self.vb_tf: self.vb,
                        self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_y0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_xb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        self.dummy_yb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        }




    def fwd_gradients_x0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]
    
    def fwd_gradients_xb(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_xb_tf)[0]
        return tf.gradients(g, self.dummy_xb_tf)[0]
        
    def fwd_gradients_y0(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_y0_tf)[0]
        return tf.gradients(g, self.dummy_y0_tf)[0]

    def fwd_gradients_yb(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_yb_tf)[0]
        return tf.gradients(g, self.dummy_yb_tf)[0]

    def preprocess(self, X):
        #H = X
        H = 2.0*(X - 0)/(0.5 - 0.0) - 1.0
        #H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0
        #H = 2.0 * (X - bound_l) / (bound_u - bound_l) - 1.0
        return H


class PINN_Burgers2(PINN): # Initialize the class 
    def __init__(self, neural_net, dt, cond_b, cond_0, param_pde, q): 
        #super(PINN_Burgers2, self).init(neural_net[0], dt, cond_b, cond_0, param_pde, q) 
        super(PINN_Burgers2, self).init(neural_net[0], dt, q)

        #self.q = q
        self.Re = param_pde[0]
        #self.NN = neural_net[0]
        self.NN2 = neural_net[1]
    
        self.cond_b = cond_b
        self.cond_i = cond_0
    
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                 log_device_placement=True))
    
        self.x0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.y0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.xb_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.yb_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_y0_tf = tf.placeholder(tf.float32, shape=(None, self.q))
        self.dummy_xb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))
        self.dummy_yb_tf = tf.placeholder(tf.float32, shape=(None, self.q+1))

        self.u0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.ub_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.v0_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.vb_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.U0_pde, self.V0_pde = self.net_U0(self.x0_tf, self.y0_tf) # N x (q+1)
        self.Ub_pred , self.Vb_pred = self.net_U1(self.xb_tf, self.yb_tf) # 2 x (q+1)


        # PDE Condition + initial condition + Boundary Condition
        self.loss = tf.reduce_sum(tf.square(self.U0_pde - self.u0_tf))
        self.loss += tf.reduce_sum(tf.square(self.V0_pde - self.v0_tf))
        self.loss += tf.reduce_sum(tf.square(self.Ub_pred - self.ub_tf))
        self.loss += tf.reduce_sum(tf.square(self.Vb_pred - self.vb_tf))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        #lr = 1e-4
        #lr = 1e-3
        lr = 4e-3
        self.optimizer_Adam = tf.train.AdamOptimizer(lr)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    

    def net_U0(self, x, y):

        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
    
        U_net = self.NN.forward(X)
        U = U_net[:, :-1]# (N, q+1) -> (N, q)
        U_x = self.fwd_gradients_x0(U, x)
        U_y = self.fwd_gradients_y0(U, y)
        U_xx = self.fwd_gradients_x0(U_x, x)
        U_yy = self.fwd_gradients_y0(U_y, y)
    
        V_net = self.NN2.forward(X)
        V = V_net[:, :-1]# (N, q+1) -> (N, q)
        V_x = self.fwd_gradients_x0(V, x)
        V_y = self.fwd_gradients_y0(V, y)
        V_xx = self.fwd_gradients_x0(V_x, x)
        V_yy = self.fwd_gradients_y0(V_y, y)

        F_U = - (U * U_x + V * U_y) + (U_xx + U_yy) / self.Re
        F_V = - (U * V_x + V * V_y) + (V_xx + V_yy) / self.Re

        #prediction of U0 using Runge-Kutta
        U0_pde = U_net - self.dt * tf.matmul(F_U, self.IRK_weights.T)# (N, q) * (q+1, q)T -> (N, q+1)
        V0_pde = V_net - self.dt * tf.matmul(F_V, self.IRK_weights.T)# (N, q) * (q+1, q)T -> (N, q+1)
        return U0_pde, V0_pde


    def net_U1(self, x, y):
        X = tf.concat([x, y], 1)
        X = self.preprocess(X)
    
        U1 = self.NN.forward(X)
        V1 = self.NN2.forward(X)
    
        return U1, V1 # N x (q+1)


    def predict(self, x_star, y_star):
    
        U, V = self.sess.run(self.Ub_pred, self.Vb_pred, {self.xb_tf: x_star, self.yb_tf: y_star})
                
        return U, V
    
    
    def callback(self, loss):

        self.renew_random()
        self.renew_tf()
        print('Loss:', loss)

    def renew_random(self):

        sp_b, bound = self.cond_b.generate()
        #print(bound[0].shape)
        self.ub = bound[0][:, None]
        self.vb = bound[1][:, None]
        self.xb = sp_b[:, 0][:, None]
        self.yb = sp_b[:, 1][:, None]

        sp_0, init = self.cond_i.generate()
        self.u0 = init[0][:, None]
        self.v0 = init[1][:, None]
        self.x0 = sp_0[:, 0][:, None]
        self.y0 = sp_0[:, 1][:, None]
        
        #print(self.xb.shape, self.yb.shape)
        #print(self.ub.shape, self.vb.shape)
        #print(self.x0.shape, self.y0.shape)
        #print(self.u0.shape, self.v0.shape)
    

    def renew_tf(self):
        self.tf_dict = {self.x0_tf: self.x0, 
                        self.y0_tf: self.y0, 
                        self.u0_tf: self.u0,
                        self.v0_tf: self.v0,
                        self.xb_tf: self.xb,
                        self.yb_tf: self.yb,
                        self.ub_tf: self.ub,
                        self.vb_tf: self.vb,
                        self.dummy_x0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_y0_tf: np.ones((self.x0.shape[0], self.q)),
                        self.dummy_xb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        self.dummy_yb_tf: np.ones((self.xb.shape[0], self.q+1)),
                        }




    def fwd_gradients_x0(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_x0_tf)[0]
        return tf.gradients(g, self.dummy_x0_tf)[0]

    def fwd_gradients_xb(self, U, x):        
        g = tf.gradients(U, x, grad_ys=self.dummy_xb_tf)[0]
        return tf.gradients(g, self.dummy_xb_tf)[0]
        
    def fwd_gradients_y0(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_y0_tf)[0]
        return tf.gradients(g, self.dummy_y0_tf)[0]

    def fwd_gradients_yb(self, U, y):        
        g = tf.gradients(U, y, grad_ys=self.dummy_yb_tf)[0]
        return tf.gradients(g, self.dummy_yb_tf)[0]

    def preprocess(self, X):
        H = 2.0*(X - 0)/(0.5 - 0.0) - 1.0
        return H






