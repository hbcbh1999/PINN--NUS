import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
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
 
def callback(loss):
    print('Loss: %.3e' % (loss))
      
#class PhysicsInformedNN:
    # Initialize the class
    #def __init__(self, P_back, x, P, layers):       
        
    #def xavier_init(self, size):
       
    
    #def neural_net(self, X, weights, biases):
        
        
    #def net_NS(self, P_back, x):
        
        
        #del_P = del_and_sh[:,0:1]
        #shock_loc = del_and_sh[:,1:2]
        
        #return P
    
    #def callback(self, loss):
        
      
    #def train(self, nIter): 

        
            
    
    #def predict(self, P_back_test, x_test):
        
        
        
    #return P_test

if __name__ == "__main__": 
      
    N_train = 1260
    
    layers = [2, 10, 10, 1]
    
    # Load Data
    data = np.loadtxt('cdnozzle1.txt')
    
    A = np.random.choice(range(data.shape[0]), size=(N_train,), replace=False)

    P_back_train = data[A,0:1].flatten()[:,None]
    x_train = data[A,1:2].flatten()[:,None]
    P_train = data[A,2:3].flatten()[:,None]

    # Training
    #model = PhysicsInformedNN(P_back_train, x_train, P_train, layers)
    X1 = np.concatenate([P_back_train, x_train], 1)
    lb = X1.min(0)
    ub = X1.max(0)
        
    # Initialize NN
    #self.weights, self.biases = self.initialize_NN(layers)
                       
    # tf placeholders and graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

    P_back_tf = tf.placeholder(tf.float32, shape=[None, P_back_train.shape[1]])
    x_tf = tf.placeholder(tf.float32, shape=[None, x_train.shape[1]])
        
    P_tf = tf.placeholder(tf.float32, shape=[None, P_train.shape[1]])
        
    #P_pred = self.net_NS(self.P_back_tf, self.x_tf)
    #P_pred = self.neural_net(tf.concat([P_back, x], 1), self.weights, self.biases)

    X = tf.concat([P_back_tf, x_tf], 1) 

    weights = []
    biases = []
    num_layers = len(layers) 

    for l in range(0,num_layers-1):
        size=[layers[l], layers[l+1]]
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
        

    num_layers = len(weights) + 1
        
    H = 2.0*(X - lb)/(ub - lb) - 1.0
    tf.cast(H, tf.float32)
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    P_pred = tf.add(tf.matmul(H, W), b)
    
        
    loss = tf.reduce_sum(tf.square(P_tf - P_pred)) 
                    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 150000,
                                                                           'maxfun': 150000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
    optimizer_Adam = tf.train.AdamOptimizer()
    train_op_Adam = optimizer_Adam.minimize(loss)                    
        
    init = tf.global_variables_initializer()
    sess.run(init)


    #model.train(10000)

    tf_dict = {P_back_tf: P_back_train, 
                    x_tf: x_train, P_tf: P_train}
        
    start_time = time.time()
    for it in range(10000):
        sess.run(train_op_Adam, tf_dict)
            
        # Print
        if it % 10 == 0:
            elapsed = time.time() - start_time
            loss_value = sess.run(loss, tf_dict)
                
            print('It: %d, Loss: %.3e, Time: %.2f' % 
                    (it, loss_value, elapsed))
            start_time = time.time()
            
    optimizer.minimize(sess,
                            feed_dict = tf_dict,
                            fetches = [loss],
                            loss_callback = callback)
      
    # Test Data
    #data1 = data
    #data1 = np.delete(data1, A, 0)
    data_test = np.loadtxt('cdnozzle_test.txt')

    #P_back_test = data1[:,0:1].flatten()[:,None]
    #x_test = data1[:,1:2].flatten()[:,None]
    for i in range(0, 8):
        P_back_test = 0.01*(27+8*i)*np.ones((100,1), dtype = float)
        x_test = 0.03*np.arange(1, 101, dtype=float).flatten()[:,None]
        P_test = data_test[100*i:100*(i+1),2:3]

    #P_test = data1[:,2:3].flatten()[:,None]
    
    # Prediction
        #P_pred = model.predict(P_back_test, x_test)
        tf_dict_test = {P_back_tf: P_back_test, x_tf: x_test}
        
        P_pred_test = sess.run(P_pred, tf_dict_test)

        plt.plot(x_test, P_pred_test, 'r--')
        plt.plot(x_test, P_test, 'b--')
    #print(P_test_pred)
    
    #Error
        error_P = np.linalg.norm(P_test-P_pred_test,2)/np.linalg.norm(P_test,2)
        print('Error P: %e' % (error_P))  

    
    plt.show()
    plt.title('Comparison of Neural Network and CFD')
    plt.xlabel('Length of nozzle')
    plt.ylabel('Pressure')
    
    # Plot Results
#    plot_solution(x_test, P_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)