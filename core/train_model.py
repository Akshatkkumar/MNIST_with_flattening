import NN_architecture as ai
import matplotlib.pyplot as plt 
import MNIST_with_flattening.core.fractal_flatten as fl
import numpy as np

# Training model
print("Data processing")
X, Y = ai.mnistgetdata(N=52900)
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

