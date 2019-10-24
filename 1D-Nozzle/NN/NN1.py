"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
tf.set_random_seed(1234)

class NN:
    # Initialize the class
    def __init__(self, P_back, x, P, U1, U2, U3, layers):
        
        X = np.concatenate([P_back, x], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.P_back = P_back
        self.x = x
        self.P = P
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.P_back_tf = tf.placeholder(tf.float32, shape=[None, self.P_back.shape[1]])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.P_tf = tf.placeholder(tf.float32, shape=[None, self.P.shape[1]])
        self.U1_tf = tf.placeholder(tf.float32, shape=[None, self.U1.shape[1]])
        self.U2_tf = tf.placeholder(tf.float32, shape=[None, self.U2.shape[1]])
        self.U3_tf = tf.placeholder(tf.float32, shape=[None, self.U3.shape[1]])
        
        self.P_pred, self.U1_pred, self.U2_pred, self.U3_pred, self.mass_flow_grad_pred, \
        self.momentum_grad_pred, self.energy_grad_pred = self.net_NS(self.P_back_tf, self.x_tf)
        self.state_eq = self.P_pred - self.P_tf

        w = 40
        self.loss = tf.reduce_sum(tf.square(self.P_tf - self.P_pred)) + \
                    tf.reduce_sum(tf.square(self.U1_tf - self.U1_pred)) + \
                    tf.reduce_sum(tf.square(self.U2_tf - self.U2_pred)) + \
                    tf.reduce_sum(tf.square(self.U3_tf - self.U3_pred)) + \
                    w*tf.reduce_sum(tf.square(self.mass_flow_grad_pred)) + \
                    w*tf.reduce_sum(tf.square(self.energy_grad_pred)) + \
                    w*tf.reduce_sum(tf.square(self.momentum_grad_pred)) + \
                    w*tf.reduce_sum(tf.square(self.state_eq))
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 15000,
                                                                           'maxfun': 15000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    # neural operation for output
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, P_back, x):
        P_U1_U2_U3 = self.neural_net(tf.concat([P_back, x], 1), self.weights, self.biases)
        P = P_U1_U2_U3[:,0:1]
        U1 = P_U1_U2_U3[:,1:2]
        U2 = P_U1_U2_U3[:,2:3]
        U3 = P_U1_U2_U3[:,3:4]

        # autodiff gradient #1
        mass_flow_grad = tf.gradients(U2, x)[0]
        # autodiff gradient #2
        S = 1 + 2.2*(x-1.5)*(x-1.5)
        momentum_grad = tf.add(tf.gradients(U2*U2/U1 + P*S, x)[0], - tf.multiply(tf.gradients(S, x)[0], P))
        # autodiff gradient #3
        rho_u_E_S = tf.divide(tf.multiply(U2, U3), U1)
        p_u_S = tf.divide(tf.multiply(tf.multiply(P, S), U2), U1)
        net_energy_expression = tf.add(rho_u_E_S, p_u_S)
        energy_grad = tf.gradients(net_energy_expression, x)[0]

        return P, U1, U2, U3, mass_flow_grad, momentum_grad, energy_grad
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
      
    def train(self, nIter): 

        tf_dict = {self.P_back_tf: self.P_back, self.x_tf: self.x,
                    self.P_tf: self.P, self.U1_tf: self.U1, self.U2_tf: self.U2, self.U3_tf: self.U3
                    }
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
            
    
    def predict(self, P_back_test, x_test):
        tf_dict = {self.P_back_tf: P_back_test, self.x_tf: x_test}
        P_test = self.sess.run(self.P_pred, tf_dict)
        U1_test = self.sess.run(self.U1_pred, tf_dict)
        U2_test = self.sess.run(self.U2_pred, tf_dict)
        U3_test = self.sess.run(self.U3_pred, tf_dict)
        return P_test, U1_test, U2_test, U3_test

if __name__ == "__main__":
    
    layers = [2, 10, 25, 50, 25, 15, 4]
    
    # Load Data
    data = np.loadtxt('cdnozzle.txt')

    N_train = int(0.8*data.shape[0])
    
    A = np.random.choice(range(data.shape[0]), size=(N_train,), replace=False)

    # x
    P_back_train = data[A,1:2].flatten()[:,None]
    x_train = data[A,0:1].flatten()[:,None]
    # y
    P_train = data[A,2:3].flatten()[:,None]
    U1_train = data[A,3:4].flatten()[:,None]
    U2_train = data[A,4:5].flatten()[:,None]
    U3_train = data[A,5:6].flatten()[:,None]

    # Training
    model = NN(P_back_train, x_train, P_train, U1_train, U2_train, U3_train, layers)
    model.train(1000)
    
    # Test Data
    data1 = data
    data1 = np.delete(data1, A, 0)

    # x
    P_back_test = data1[:,1:2].flatten()[:,None]
    x_test = data1[:,0:1].flatten()[:,None]
    # y
    P_test = data1[:,2:3].flatten()[:,None]
    U1_test = data1[:,3:4].flatten()[:,None]
    U2_test = data1[:,4:5].flatten()[:,None]
    U3_test = data1[:,5:6].flatten()[:,None]

    # Prediction
    P_pred, U1_pred, U2_pred, U3_pred  = model.predict(P_back_test, x_test)
    
    # Error
    error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
    print("Test Error in P: "+str(error_P))
    error_U1 = np.linalg.norm(U1_test-U1_pred,2)/np.linalg.norm(U1_test,2)
    print("Test Error in U1: "+str(error_U1))
    error_U2 = np.linalg.norm(U2_test-U2_pred,2)/np.linalg.norm(U2_test,2)
    print("Test Error in U2: "+str(error_U2))
    error_U3 = np.linalg.norm(U3_test-U3_pred,2)/np.linalg.norm(U3_test,2)
    print("Test Error in U3: "+str(error_U3))

    #Plotting
    plt_l=700
    plt_u=800
    # x
    x_test_plt = data[plt_l:plt_u, 0:1].flatten()[:,None]
    P_back_test_plt = data[plt_l:plt_u, 1:2].flatten()[:,None]
    # y
    P_test_plt = data[plt_l:plt_u, 2:3].flatten()[:,None]
    U1_test_plt = data[plt_l:plt_u, 3:4].flatten()[:,None]
    U2_test_plt = data[plt_l:plt_u, 4:5].flatten()[:,None]
    U3_test_plt = data[plt_l:plt_u, 5:6].flatten()[:,None]
    # predict
    P_pred_plt, U1_pred_plt, U2_pred_plt, U3_pred_plt = model.predict(P_back_test_plt, x_test_plt)
    # plot P
    plt.plot(x_test_plt, P_pred_plt, 'ro', label='NN')
    plt.plot(x_test_plt, P_test_plt, 'g--', label='CFD')
    plt.legend()
    plt.title('Comparison of Neural Network and CFD')
    plt.xlabel('Length of nozzle')
    plt.ylabel('Pressure*')
    plt.show()
    # plot U1
    plt.plot(x_test_plt, U1_pred_plt, 'ro', label='NN')
    plt.plot(x_test_plt, U1_test_plt, 'g--', label='CFD')
    plt.legend()
    plt.title('Comparison of Neural Network and CFD')
    plt.xlabel('Length of nozzle')
    plt.ylabel('rho_S')
    plt.show()
    # plot U2
    plt.plot(x_test_plt, U2_pred_plt, 'ro', label='NN')
    plt.plot(x_test_plt, U2_test_plt, 'g--', label='CFD')
    plt.legend()
    plt.title('Comparison of Neural Network and CFD')
    plt.xlabel('Length of nozzle')
    plt.ylabel('rho_u_S')
    plt.show()
    # plot U3
    plt.plot(x_test_plt, U3_pred_plt, 'ro', label='NN')
    plt.plot(x_test_plt, U3_test_plt, 'g--', label='CFD')
    plt.legend()
    plt.title('Comparison of Neural Network and CFD')
    plt.xlabel('Length of nozzle')
    plt.ylabel('rho_E_S')
    plt.show()