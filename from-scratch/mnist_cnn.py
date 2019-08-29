from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pylab as plt
from cnn_functions import train_nn, predict_y
from sklearn.metrics import accuracy_score

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

# Train the network
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)
# plt.plot(avg_cost_func)
# plt.ylabel('Average J')
# plt.xlabel('Iteration number')
# plt.show()

y_pred = predict_y(W, b, X_test, 3)
print(accuracy_score(y_test, y_pred)*100)
