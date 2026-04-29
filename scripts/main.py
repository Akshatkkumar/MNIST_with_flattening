# loading and using model
from core import NN_architecture as ai
import matplotlib.pyplot as plt
from core import fractal_flatten as fl
import numpy as np

print("RUNNING TEST ON RANDOM DIGIT FROM MNIST")
print("_______________________________________")
X, Y = ai.mnistgetdata()

Ws, bs, fs = ai.load_model("model")

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
index = np.random.choice(list(range(1000)))
mapp = fl.H_Lo(28,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print("DISPLAYING FRACTAL CURVE USED")
print("______________________________")
fl.plotmapping(mapp, nums_visible=False)
print("DISPLAYING RESULTS ON TEST IMAGE")
vis(X[index], lambda l: ai.model(fl.flatten_gen(l, mapp), Ws, bs, fs)[-1][-1], show=True)
