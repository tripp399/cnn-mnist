import numpy as np

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

def calculate_out_layer_delta(y, h_out, z_out):
  """
  calculation of output layer delta to perform backpropagation
  """
  return -(y-h_out) * g(z_out)

def calculate_hidden_layer_delta(delta_plus_1, w_l, z_l):
  """
  calculation of hidden layer delta to perform backpropagation
  """
  return np.dot(np.transpose(w_l), delta_plus_1) * g(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
  W, b = setup_and_init_weights(nn_structure)
  cnt = 0
  m = len(y)
  avg_cost_func = []
  print('Starting gradient descent for {} iterations'.format(iter_num))
  while cnt < iter_num:
    if cnt%100 == 0:
      print('Iteration {} of {}'.format(cnt, iter_num))
    tri_W, tri_b = init_tri_values(nn_structure)
    avg_cost = 0
    for i in range(len(y)):
      delta = {}
      # perform the feed forward pass and return the stored h and z values, to be used in the
      # gradient descent step
      h, z = feed_forward(X[i, :], W, b)
      # loop from nl-1 to 1 backpropagating the errors
      for l in range(len(nn_structure), 0, -1):
        if l == len(nn_structure):
          delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
          avg_cost += np.linalg.norm((y[i,:]-h[l]))
        else:
          if l > 1:
            delta[l] = calculate_hidden_layer_delta(delta[l+1], W[l], z[l])
          # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
          tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
          # trib^(l) = trib^(l) + delta^(l+1)
          tri_b[l] += delta[l+1]
    # perform the gradient descent step for the weights in each layer
    for l in range(len(nn_structure) - 1, 0, -1):
      W[l] += -alpha * (1.0/m * tri_W[l])
      b[l] += -alpha * (1.0/m * tri_b[l])
    # complete the average cost calculation
    avg_cost = 1.0/m * avg_cost
    avg_cost_func.append(avg_cost)
    cnt += 1
  return W, b, avg_cost_func