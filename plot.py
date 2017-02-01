import numpy as np
import matplotlib.pyplot as plt

# load the data to plot
data = np.loadtxt("./data.dat")

# get the weights by running the executable
weights = np.array([-7.196486234790699, 1.4995485998451574, 2.3699539634435407])

# create a separation line
xx = np.linspace(np.min(data[:, 0])-1, np.max(data[:, 0])+1, 10)
yy = -weights[0]/weights[2]-weights[1]/weights[2]*xx

# plot the data and separation
plt.plot(data[0:100,0], data[0:100,1], 'bo',label='class 0')
plt.plot(data[100:200,0], data[100:200,1], 'go', label='class 1')
plt.plot(xx,yy,'r-', label='decision boundary')
plt.xlim(np.min(data[:, 0])-1, np.max(data[:, 0])+1)
plt.legend(loc=0)
plt.savefig('./single_neuron.pdf')
