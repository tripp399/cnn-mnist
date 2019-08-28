from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()

# check dataset
# print(digits.data.shape)
# import matplotlib.pyplot as plt 
# plt.gray() 
# plt.matshow(digits.images[1]) 
# plt.show()
# print(digits.data[0, :])

# pre-processing
scaler = StandardScaler()
X = scaler.fit_transform(digits.data)
y = digits.target
# print(X[0, :])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

def convert_y_to_vector(y):
  vector = np.zeros((len(y), 10))
  for i in range(len(y)):
    vector[i, y[i]] = 1
  return vector

y_v_train = convert_y_to_vector(y_train)
y_v_test = convert_y_to_vector(y_test)
# print(y_train[0], y_v_train[0])

# Define number of nodes in each layer of neural network
# 64 input nodes for each pixel in the image
# and 10 output nodes for 0-9 possible values
nn_structure = [64, 30, 10]

def f(x):
  """
  sigmoid activation function
  """
  return 1 / (1 + np.exp(-x))

def g(x):
  """
  gradient funtion for sigmoid
  """
  return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
  """
  initialise weights and bias for each layer with random values
  """
  W = {}
  b = {}
  for l in range(1, len(nn_structure)):
    W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1]))
    b[l] = np.random.random_sample((nn_structure[l],))
  return W, b

def init_tri_values(nn_structure):
  """
  initialise mean accumulation value for each layer
  """
  tri_W = {}
  tri_b = {}
  for l in range(1, len(nn_structure)):
    tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
    tri_b[l] = np.zeros((nn_structure[l],))
  return tri_W, tri_b

def feed_forward(x, W, b):
  """
  feed forward pass though the network for input vector x
  """
  h = {1: x}
  z = {}
  for l in range(1, len(W) + 1):
      if l == 1:
        node_in = x
      else:
        node_in = h[l]
      z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
      h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
  return h, z