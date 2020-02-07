
import sys
sys.path.insert(0, 'Utilities/')
import tensorflow as tf 
import numpy as np 
import scipy.io 
import sampling 
import time

class PINN_c(object):

    # Initialize the class
    def __init__(self, neural_net):

        # Initialize NN
        #self.NN = self.NeuralNet(layers)
        self.NN = neural_net
    
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

class PINN_2d(PINN_c):

    # Initialize the class 
    def __init__(self, neural_net, cond_b, param_pde):

        super(PINN_2d, self).__init__(neural_net)
    
        #self.name = 'disffusion'
        self.path = "model//diff2d.ckpt"

        self.cond_b = cond_b
        #self.bound_l = self.cond_0.lower[0]
        #self.bound_u = self.cond_0.upper[0]
    
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




