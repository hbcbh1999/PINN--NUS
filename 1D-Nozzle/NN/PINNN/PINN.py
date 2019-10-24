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
import csv

np.random.seed(1234)
tf.set_random_seed(1234)

class PINN:
    # Initialize the class
    def __init__(self, P_back, x, P, rho, u, E, layers):
        
        X = np.concatenate([P_back, x], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.X = X
        self.P_back = P_back
        self.x = x
        self.P = P
        self.rho = rho
        self.u = u
        self.E = E

        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.P_back_tf = tf.placeholder(tf.float32, shape=[None, self.P_back.shape[1]])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.P_tf = tf.placeholder(tf.float32, shape=[None, self.P.shape[1]])
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.E_tf = tf.placeholder(tf.float32, shape=[None, self.E.shape[1]])
        
        self.P_pred, self.rho_pred, self.u_pred, self.E_pred, self.e1, self.e2, self.e3, self.e4 = self.net_NS(self.P_back_tf, self.x_tf)

        # MSE Normalization
        P_norm = np.amax(P)
        rho_norm = np.amax(rho)
        u_norm = np.amax(u)
        E_norm = np.amax(E)
        S_norm = 5.95
        e1_norm = rho_norm*u_norm*S_norm
        e2_norm = P_norm*S_norm
        e3_norm = E_norm*rho_norm*u_norm*S_norm
        a = 40

        self.loss = tf.reduce_sum(tf.square(self.P_tf - self.P_pred))/(P_norm**2) + \
                    tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2) + \
                    tf.reduce_sum(tf.square(self.u_tf - self.u_pred))/(u_norm**2) + \
                    tf.reduce_sum(tf.square(self.E_tf - self.E_pred))/(E_norm**2) + \
                    a*tf.reduce_sum(tf.square(self.e2))/(e2_norm**2) + \
                    a*tf.reduce_sum(tf.square(self.e3))/(e3_norm**2) + \
                    a*tf.reduce_sum(tf.square(self.e1))/(e1_norm**2) + \
                    a*tf.reduce_sum(tf.square(self.e4))/(P_norm**2)
                    
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
        P_rho_u_E = self.neural_net(tf.concat([P_back, x], 1), self.weights, self.biases)
        P = P_rho_u_E[:,0:1]
        rho = P_rho_u_E[:,1:2]
        u = P_rho_u_E[:,2:3]
        E = P_rho_u_E[:,3:4]

        S = 1 + 2.2*(3*x-1.5)**2
        # autodiff gradient #1
        mass_flow_grad = tf.gradients(rho*u*S, x)[0]
        # autodiff gradient #2
        momentum_grad = tf.gradients((rho*u*u + P)*S, x)[0] - P*tf.gradients(S, x)[0]
        # autodiff gradient #3
        energy_grad = tf.gradients((rho*E + P)*u*S, x)[0]
        # state residual
        gamma = 1.4
        state_res = P - rho*(gamma-1)*(E-0.5*gamma*u*u)

        return P, rho, u, E, mass_flow_grad, momentum_grad, energy_grad, state_res
    
    def callback(self, loss):
        loss_vector.append(loss)
        print('Loss: %.3e' % (loss))
      
    def train(self, nIter): 

        tf_dict = {self.P_back_tf: self.P_back, self.x_tf: self.x,
                    self.P_tf: self.P, self.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E
                    }
        
        global loss_vector
        loss_vector = []

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            loss_value = self.sess.run(self.loss, tf_dict)

            loss_vector.append(loss_value)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                # res1 = self.sess.run(self.e1, tf_dict)
                # res2 = self.sess.run(self.e2, tf_dict)
                # res3 = self.sess.run(self.e3, tf_dict)
                print('Iter: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                # print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                #     %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
                start_time = time.time()
            
        # self.optimizer.minimize(self.sess,
        #                        feed_dict = tf_dict,
        #                        fetches = [self.loss],
        #                        loss_callback = self.callback)

        return loss_vector
            
    
    def predict(self, P_back_test, x_test):
        tf_dict = {self.P_back_tf: P_back_test, self.x_tf: x_test}
        P_test = self.sess.run(self.P_pred, tf_dict)
        rho_test = self.sess.run(self.rho_pred, tf_dict)
        u_test = self.sess.run(self.u_pred, tf_dict)
        E_test = self.sess.run(self.E_pred, tf_dict)
        return P_test, rho_test, u_test, E_test


layers = [2, 10, 10, 10, 4]

P=[]
rho=[]
E=[]
u=[]

with open('cdn_P.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            P.append(float(e))

with open('cdn_rho.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            rho.append(float(e))

with open('cdn_E.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            E.append(float(e))

with open('cdn_u.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            u.append(float(e))

P = np.asarray(P)
rho = np.asarray(rho)
E = np.asarray(E)
u = np.asarray(u)

pb=[]
pb=np.asarray(pb)

z=[]
z=np.asarray(z)

for i in range(0, 27):
    P_back = 0.01*(21+3*i)*np.ones((101,1))
    pb = np.concatenate((pb, P_back), axis=None)
    x_l = 0.01*np.arange(0, 101, dtype=float).flatten()[:,None]
    z = np.concatenate((z, x_l), axis=None)

train_frac = 0.01
N_train = int(train_frac*P.shape[0])

A = np.random.choice(range(P.shape[0]), size=(N_train,), replace=False)

# x
P_back_train = pb[A].flatten()[:,None]
x_train = z[A].flatten()[:,None]
# y
P_train = P[A].flatten()[:,None]
rho_train = rho[A].flatten()[:,None]
u_train = u[A].flatten()[:,None]
E_train = E[A].flatten()[:,None]

# Training
model = PINN(P_back_train, x_train, P_train, rho_train, u_train, E_train, layers)
model.train(20001)

# plt.plot(loss_vector, label='Loss value')
# print("Total Iter = " + str(len(loss_vector)))
# plt.legend()
# plt.title('Loss value over iterations')
# plt.xlabel('#iterations')
# plt.ylabel('Loss')
# plt.show()

# Test Data
P_test=[]
rho_test=[]
E_test=[]
u_test=[]

with open('cdn_P_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            P_test.append(float(e))

with open('cdn_rho_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            rho_test.append(float(e))

with open('cdn_E_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            E_test.append(float(e))

with open('cdn_u_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        for e in row:
            u_test.append(float(e))

P_test = np.asarray(P_test).flatten()[:,None]
rho_test = np.asarray(rho_test).flatten()[:,None]
E_test = np.asarray(E_test).flatten()[:,None]
u_test = np.asarray(u_test).flatten()[:,None]


pb_test=[]
pb_test=np.asarray(pb_test)

z_test=[]
z_test=np.asarray(z_test)

for i in range(0, 6):
    P_back = 0.01*(35+10*i)*np.ones((101,1))
    pb_test = np.concatenate((pb_test, P_back), axis=None)
    x_l = 0.01*np.arange(0, 101, dtype=float).flatten()[:,None]
    z_test = np.concatenate((z_test, x_l), axis=None)

pb_test = pb_test.flatten()[:,None]
z_test = z_test.flatten()[:,None]

# Prediction
P_pred, rho_pred, u_pred, E_pred  = model.predict(pb_test, z_test)

# Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_E = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
print("Test Error in E: "+str(error_E))

#Plotting
a = 2
b = 3
s = 1
val = 101*a

S = 1 + 2.2*(3*z_test-1.5)**2
# rho_u_pred = rho_pred*u_pred
# rho_E_pred = rho_pred*E_pred
# rho_u_test = rho_test*u_test
# rho_E_test = rho_test*E_test

for i in range (101*a, 101*b, 101*s):
    plt.ylim(0, 1.0)
    plt.plot(z_test[i:i+101], P_pred[i:i+101], 'c', label='PINN' if i == val else "")
    plt.plot(z_test[i:i+101], P_test[i:i+101], 'g', label='Truth' if i == val else "")
    plt.title('Comparison of PINN results for Pressure')
    plt.xlabel('Nozzle Length')
    plt.ylabel('value')
    plt.legend(loc='center left')
plt.show()

for i in range (101*a, 101*b, 101*s):
    plt.plot(z_test[i:i+101], E_pred[i:i+101], 'b', label='PINN' if i == val else "")
    plt.plot(z_test[i:i+101], E_test[i:i+101], 'g', label='Truth' if i == val else "")
    plt.title('Comparison of PINN results for Energy')
    plt.xlabel('Nozzle Length')
    plt.ylabel('value')
    plt.legend()
plt.show()

for i in range (101*a, 101*b, 101*s):
    plt.plot(z_test[i:i+101], u_pred[i:i+101], 'c', label='PINN' if i == val else "")
    plt.plot(z_test[i:i+101], u_test[i:i+101], 'g', label='Truth' if i == val else "")
    plt.title('Comparison of PINN results for speed')
    plt.xlabel('Nozzle Length')
    plt.ylabel('value')
    plt.legend()
plt.show()

for i in range (101*a, 101*b, 101*s):
    plt.plot(z_test[i:i+101], rho_pred[i:i+101], 'y', label='PINN' if i == val else "")
    plt.plot(z_test[i:i+101], rho_test[i:i+101], 'g', label='Truth' if i == val else "")
    plt.title('Comparison of PINN results for density')
    plt.xlabel('Nozzle Length')
    plt.ylabel('value')
    plt.legend()
plt.show()

# for i in range (101*a, 101*b, 101*s):
#     plt.plot(z_test[i:i+101], rho_u_pred[i:i+101], 'y', label='PINN' if i == val else "")
#     plt.plot(z_test[i:i+101], rho_u_test[i:i+101], 'g', label='Truth' if i == val else "")
#     plt.title('Comparison of PINN results for rho_U')
#     plt.xlabel('Nozzle Length')
#     plt.ylabel('value')
#     plt.legend()
# plt.show()

# for i in range (101*a, 101*b, 101*s):
#     plt.plot(z_test[i:i+101], rho_E_pred[i:i+101], 'y', label='PINN' if i == val else "")
#     plt.plot(z_test[i:i+101], rho_E_test[i:i+101], 'g', label='Truth' if i == val else "")
#     plt.title('Comparison of PINN results for rho_E')
#     plt.xlabel('Nozzle Length')
#     plt.ylabel('value')
#     plt.legend()
# plt.show()

# for i in range (101*a, 101*b, 101*s):
#     plt.ylim(0, 0.05)
#     plt.plot(z_test[i:i+101], ((P_test[i:i+101]-P_pred[i:i+101])**2 + (u_test[i:i+101]-u_pred[i:i+101])**2 +
#                                 (E_test[i:i+101]-E_pred[i:i+101])**2 + (rho_test[i:i+101]-rho_pred[i:i+101])**2)/4, 'y', label='PINN Mean square Error')
#     flux1 = (rho_pred*u_pred*S).reshape((606, ))
#     flux2 = ((rho_pred*u_pred**2+ P_pred)*S).reshape((606, ))
#     flux3 = ((rho_pred*E_pred+P_pred)*u_pred*S).reshape((606, ))
#     S = S.reshape((606, ))
#     P_pred = P_pred.reshape((606, ))
#     P_pred = P_pred.reshape((606, ))
#     P_pred = P_pred.reshape((606, ))
#     E_pred = E_pred.reshape((606, ))
#     gamma = 1.4
#     plt.plot(z_test[i:i+101], ((np.gradient(flux1[i:i+101]))**2 +
#                                 (np.gradient(flux2[i:i+101])-P_pred[i:i+101]*np.gradient(S[i:i+101]))**2 +
#                                 (np.gradient(flux3[i:i+101]))**2)/3, 'k', label='PINN Mean square residual')
#     plt.xlabel('Nozzle Length')
#     plt.ylabel('value')
#     plt.legend(loc='center left')
# plt.show()

# #ideal gas eq
# gamma = 1.4
# RHS = rho_pred*(gamma-1)*(E_pred-0.5*u_pred*gamma*u_pred)
# for i in range (101*a, 101*b, 101*s):
#     plt.plot(z_test[i:i+101], RHS[i:i+101], 'kx', label='NN')
#     plt.plot(z_test[i:i+101], P_pred[i:i+101], 'g', label='Truth')
#     plt.title('Ideal Gas Equation')
#     plt.xlabel('Nozzle Length')
#     plt.ylabel('value')
#     plt.legend()
# plt.show()