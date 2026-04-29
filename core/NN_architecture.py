## Libraries 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
import os
##

## Intro and data prep

# Save a model to a file
def save_model(Ws, bs, fs, filename):
    np.savez_compressed(f'{filename}.npz', weights=np.array(Ws,dtype=object), bias=np.array(bs, dtype=object), funcs=[func_to_string(f) for f in fs])

# Load a model from a file 
BASE_DIR = os.path.dirname(__file__)
def load_model(filename):
    with np.load(os.path.join(BASE_DIR, f'{filename}.npz'), allow_pickle=True) as data:
        Ws = list(data['weights'])
        bs = list(data['bias'])
        fs = [string_to_func(s) for s in data['funcs']]
    return Ws, bs, fs

# fits an exponential to both points, ie such that the value is y0 at x0 and y1 at x1
# returns a tuple (C, r) indicating an exponential f(x) = C*e^rx
def expofit(x0,y0,x1,y1):
    return (np.exp((x1*np.log(y0) - x0*np.log(y1))/(x1-x0)), (np.log(y1)-np.log(y0))/(x1-x0)) 

# flatten a 2d array normally
def flatten(l):
    return np.asarray(l).flatten()

# flatten an (square) array with a space filling Hilbert curve(currently just expands out to a 2^n x 2^n curve and conserves the actual dimensions present)
def flatten_adv(l):
    # Converts a 1D index 'd' to 2D coordinates (x, y) on a Hilbert curve of size n x n.
    def d2xy(n, d):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    # Helper for flipping quadrants
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            return y, x
        return x, y

    n = int(2**(np.ceil(np.log(len(l))/np.log(2))))  # Smallest power of 2 for 28x28
    mapping = [d2xy(n, i) for i in range(n*n)]

    # Filter mapping to only include points within the 28x28 bounds
    true_mapping = [(x, y) for x, y in mapping if x < len(l) and y < len(l)]
    return  [l[y][x] for x,y in true_mapping] 

# returns MNIST dataset: X is a list of 70,000 28x28 image arrays and y is a list of 1x10 one hot encoded outputs corresponding to digit
# N controls the number of samples given
# if Normalize is set to True, it will normalize all pixel values from 0-255 to 0-1
def mnistgetdata(N='all', normalize=False):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist["data"], mnist["target"]
    if N != 'all':
        X, y = X[:N], y[:N]
    if normalize:
        X = np.asarray([np.subtract(np.multiply(2,np.divide(x,255)),1) for x in X])
    X = X.reshape(-1,28,28)
    y = [[1 if int(x) == i else 0 for i in range(10)] for x in y]
    return(X, y)

# Get the iris data and change output form
def irisgetdata():
    iris = load_iris()
    return (iris.data, [[1 if x == i else 0 for i in range(3)] for x in iris.target])

# shuffles and splits the data, decides how many from the last to split on
def split(X,Y,split):
    np.random.shuffle(inds:=list(range(len(X))))
    X_train = [X[i] for i in inds[:-split]]
    Y_train = [Y[i] for i in inds[:-split]]
    X_test = [X[i] for i in inds[-split:]]
    Y_test = [Y[i] for i in inds[-split:]]
    return(X_train, X_test, Y_train, Y_test)

# batches data, assumes data already shuffled
# takes in a single array, returns an array of batch arrays 
def batch(X_train, Y_train, batch_s):
    X_batches = []
    Y_batches = []
    curr_batch_X = []
    curr_batch_Y = []
    for i in range(len(X_train)):
        curr_batch_X.append(X_train[i])
        curr_batch_Y.append(Y_train[i])
        if (i+1) % batch_s == 0:
            X_batches.append(curr_batch_X)
            Y_batches.append(curr_batch_Y)
            curr_batch_X = []
            curr_batch_Y = []
    if len(X_train) % batch_s != 0:
        X_batches.append(curr_batch_X)
        Y_batches.append(curr_batch_Y)   
    return(X_batches, Y_batches)  

# takes a list of classifier outputs(one hot encoded) and targets and will give the hit rate where the chosen classification is given by the max
# eg y_hatl = [[0.1,0.2,0.9],[0.5,0.1,0.1]], tl = [[0,0,1],[1,0,0]] -> 100% accuracy
def classifier_accuracy(y_hatl, tl):
    acc = 0
    for i in range(len(y_hatl)):
        if [1 if y_hatl[i][j] == max(y_hatl[i]) else 0 for j in range(len(y_hatl[i]))] == tl[i]:
            acc += 1
    return(acc/len(y_hatl))
##

## Activation functions

def func_to_string(f):
    if f == sig:
        return 'sig'
    if f == RELU:
        return 'RELU'
    if f == softplus:
        return 'softplus'

def string_to_func(s):
    if s == 'sig':
        return sig
    if s == 'RELU':
        return RELU
    if s == 'softplus':
        return softplus

# Apply sigmoid to an array 
# Uses branching and alternative forms to avoid blowup when x is large negative
def sig(x):
    x = np.clip(x, -100, 100)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

# Apply the derivative of sigmoid to an array
def sigp(x):
    return np.multiply(sig(x),np.subtract(1,sig(x)))

# Apply RELU to an array(leaky RELU for a!=0)
def RELU(x):
    # adjust a to get leaky RELU
    a = 0
    return [a*k if k<0 else k for k in x]

# Apply the derivative of RELU to an array
def RELUp(x):
    a = 0
    return [a if k<0 else 1 for k in x]

# Apply softmax to an array
def softplus(x):
    return np.log(np.add(1,np.exp(x)))

# Apply derivative of softmax 
def softplusp(x):
    return sig(x)

# returns the derivative function given an activation function
def return_fp(f):
    if f == sig:
        return sigp
    if f == RELU:
        return RELUp
    if f == softplus:
        return softplusp
##

## Forward pass
# Given current layer, weight matrix, bias vector,activation function -> give next layer of NN
def next(L,W,b,f):
    Z = np.add(np.dot(L, W),b)
    return Z, f(Z)
    
# Run the model given an input, weights matrices, bias vectors, and activation functions
def model(L0, Ws, bs, fs):
    Zs, Ls = [], [L0]
    for i in range(len(Ws)):
        Z_n, L_n = next(Ls[-1], Ws[i], bs[i], fs[i])
        Zs.append(Z_n)
        Ls.append(L_n)
    return(Zs,Ls)

# Generate a random weight matrix 
# K controls the range in which initial random params are generated. eg, K=1 means vals are between -0.5,0.5
def gen_randw(L0s, L1s, K=1):
    return np.subtract(np.multiply(K,np.random.random((L0s,L1s))),K/2)

# Uses Xavier initialization to avoid vanishing gradient in larger networks, user has a choice between this and normal uniform random for initialzation
def gen_xavierw(L0s, L1s):
    # L0s is n_in, L1s is n_out
    limit = np.sqrt(6 / (L0s + L1s))
    return np.random.uniform(-limit, limit, (L0s, L1s))

# Generate a random bias vector
def gen_randb(Ls, K=1):
    return np.subtract(np.multiply(K,np.random.random(Ls)),K/2)

# Squared cost function
def sq_C(Y_hat, Y):
    v = np.subtract(Y_hat, Y)
    return(np.dot(v,v))

# Categorical cross Entropy 
def CCE(Y_hat, Y):
    return sum(np.multiply(-1, np.multiply(Y, np.log(Y_hat))))

# Derivative of the Squared cost with respect to y_hat
def sq_Cp(Y_hat, Y):
    v = np.subtract(Y_hat, Y)
    return(np.multiply(2,v))

# Derivative of Categorical Cross Entropy with respect to y_hat 
def CCEp(Y_hat, Y):
    return np.multiply(-1,np.divide(Y,Y_hat))

def return_cp(c):
    if c == sq_C:
        return sq_Cp
    if c == CCE:
        return CCEp
##

## Train and Test(backprop done in the loop) 
# Train a Neural net given initial parameters and unbatched training data(learing rate has expo decay set rate to zero if no decay is desired) 
# It returns the weights, bias and costs on each epoch
# If keepbest is on, it will return the lowest Weight and Bias pair found during training, instead of the values found on the last epoch
def train(X_train, Y_train, X_test, Y_test, batch_s, Wsi, bsi, fs, cf, epochs, lri=0.01, dec=0, keepbest=False):
    X_train_batches, Y_train_batches = batch(X_train, Y_train, batch_s)
    Ds = []
    Ws = Wsi
    bs = bsi
    param_L = []
    for epoch in range(epochs):
        # This line adds exponential decay on the learning rate(change inside of exp to 0 if no decay wanted)
        lr = lri*np.exp(dec*epoch)
        for i in range(len(X_train_batches)):
            x_batch = X_train_batches[i]
            y_batch = Y_train_batches[i]
            delWs = [np.zeros(np.asarray(k).shape) for k in Wsi]
            delbs = [np.zeros(np.asarray(k).shape) for k in bsi]
            for j in range(len(x_batch)):
                x = x_batch[j]
                y = y_batch[j]
                L0 = x
                Zs, Ls = model(L0, Ws, bs, fs)
                prod = return_cp(cf)(Ls[-1],y)
                for idx in range(len(Ws)-1,-1,-1):
                    delWs[idx] = np.add(delWs[idx], np.outer(Ls[idx],np.multiply(prod, return_fp(fs[idx])(Zs[idx]))))
                    delbs[idx] = np.add(delbs[idx], np.multiply(prod, return_fp(fs[idx])(Zs[idx])))
                    prod = np.dot(np.multiply(prod, return_fp(fs[idx])(Zs[idx])),np.transpose(Ws[idx]))
            delWs = [np.divide(del_wi, len(x_batch)) for del_wi in delWs]
            delbs = [np.divide(del_bi, len(x_batch)) for del_bi in delbs]
            for j in range(len(Ws)):
                Ws[j] = np.add(Ws[j], np.multiply(-lr,delWs[j]))
                bs[j] = np.add(bs[j], np.multiply(-lr,delbs[j]))
        param_L.append((Ws,bs))
    
        D = 0
        for i in range(len(X_test)):
            x_t = X_test[i]
            y_t = Y_test[i]
            Zts, Lts = model(x_t,Ws,bs,fs)
            D = D + cf(Lts[-1], y_t)
        D = D/len(X_test)
        print(f'Current Cost on Testing: {D}')
        Ds.append(D)
    
    if keepbest:
        Ws, bs = param_L[np.argmax(Ds)]
        return Ws, bs, Ds[:np.argmax(Ds)]
    return Ws, bs, Ds


# function for testing a trained model on given data
def test(X_test, Y_test, Ws, bs, fs, cf, show = False):
    D = 0
    for i in range(len(X_test)):
        x_t = X_test[i]
        y_t = Y_test[i]
        Zts, Lts = model(x_t,Ws,bs,fs)
        D = D + (cv:=cf(Lts[-1], y_t))
        if show:
            print(f'Predicted output {y_t} | Actual output {Lts[-1]}')
            print(f'Cost: {cv}')
    D = D/len(X_test)
    print(f'Average cost on testing: {D}')