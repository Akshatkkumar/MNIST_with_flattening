import ailib_opt as ai
import matplotlib.pyplot as plt 
import flatten_lib as fl
from PIL import Image
import numpy as np

# Training model
print("Data processing")
X, Y = ai.mnistgetdata(N=52900)
# adv flattening used on mnist784-300-20-10 versions 1-12
# S flattening used on mnist784-300-20-10 version 12,13,14,15,16,17 S 
# Z flattening used on mnist784-300-20-10 version 12 Z
# H-L flattening used on mnist784-300-20-10 version 17, 18, 19, 20 HL
# H-Lo flattening used on mnist784-300-20-10 version 21, 22 HLo. The L_ors are given below
# 21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 23: [1]*80
# 24: [0]*35 + [1]*10 + [0]*35
print("flattening")
mapp = fl.H_Lo(28,[0]*80)
X = [fl.flatten_gen(x, mapp) for x in X]
print(len(X), len(Y))
test_s = 1000
batch_s = 30
X_train, X_test, Y_train, Y_test = ai.split(X,Y,test_s)
Ws = [ai.gen_xavierw(784,360), ai.gen_xavierw(360,180), ai.gen_xavierw(180,30), ai.gen_xavierw(30,10)]
bs = [ai.gen_randb(360), ai.gen_randb(180), ai.gen_randb(30),ai.gen_randb(10)]
fs = [ai.RELU, ai.sig, ai.RELU, ai.sig]
costfunc = ai.sq_C
Ws, bs, costs = ai.train(X_train, Y_train, X_test, Y_test, batch_s, Ws, bs, fs, costfunc, 75, lri=0.033, dec=-0.013488012155713065, keepbest=True)
ai.test(X_test, Y_test, Ws, bs, fs, costfunc, show = True)
plt.plot(costs)
plt.show()
print(f'accuracy: {ai.classifier_accuracy([ai.model(x, Ws, bs, fs)[-1][-1] for x in X_test], Y_test)}')
ai.save_model(Ws,bs,fs,"mnist_784_300_20_10_ver24_HLo")
# 0.45 Z, 0.38 S, 0.35 Hil_fil
#

def conv(filename):
    img = Image.open(filename).convert('L')
    img = img.resize((28,28))
    return(np.subtract(255, np.asarray(img).astype('float32')))

# Model usage example
'''
# loading and using model
import ailib_opt as ai
import matplotlib.pyplot as plt
import flatten_lib as fl
Ws, bs, fs = ai.load_model("mnist_784_300_20_10_ver23_HLo")
X, Y = ai.mnistgetdata()
# Allow us to visualize a given 28x28 grid as an image
# if given a model function(That is, a lambda with preloaded weights, bias, funcs, flattening) 
# it will also plot a bar plot of the models confidence in each number
def vis(arr, modelfunc=None, show=False):
    fig, axs = plt.subplots(1,2,figsize=(13,6))
    axs[0].imshow(arr, cmap='gray')
    axs[1].bar(['0','1','2','3','4','5','6','7','8','9'],modelfunc(arr))
    if show:
        plt.show()

# make sure to adjust flattening mode as needed
mapp = fl.H_Lo(28,[1]*80)
vis(X[1023], lambda l: ai.model(fl.flatten_gen(l, mapp), Ws, bs, fs)[-1][-1], show=True)
ai.classifier_accuracy([ai.model(fl.flatten_gen(x, mapp), Ws, bs, fs)[-1][-1] for x in X], Y)
'''