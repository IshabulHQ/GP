#imports for plotting Gaussian Process
import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')
np.random.seed(42)


def polynomial_cov(xi,xj):
    return (np.dot(xi,xj.transpose())+ 0.1)
# Sample from the Gaussian process distribution
nb_of_samples = 50  # Number of points in each function
number_of_functions = 5  # Number of functions to sample

# Independent variable samples
X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
Σ = polynomial_cov(X,X)  # Kernel of data points


# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=Σ,
    size=number_of_functions)

# Plot the sampled functions
plt.figure(figsize=(6, 4))
for i in range(number_of_functions):
    plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$x$', fontsize=13)
plt.ylabel('$y = f(x)$', fontsize=13)
plt.title((
    'Polynomial covariance function'))
plt.xlim([-4, 4])
plt.show()
#
