import numpy as np
import sklearn
import matplotlib.pyplot as plt

import sklearn.datasets
def sigmoid(z) :
    return 1.0/(1.0+np.exp(-z))

def h_from_x(X,theta) :
    #print("X shape " + str(X.shape))
    z = np.dot(X,theta)
    h = sigmoid(z)
    return h

def loss(h,y) :
    l = (-y) * np.log(h) - (1.0-y)*np.log(1.0-h)
    return l.mean()

def grad(X,h,y) :

    return np.dot(X.T,(h-y))/y.shape[0]

def learn_theta(theta,gradient,lr=0.01) :
    return theta-(lr*gradient)


def fit(X,Y):
    #add ones here
    #intercept = np.ones((X.shape[0], 1))
    #X = np.concatenate((intercept, X), axis=1)
    print(X)
    print(Y.shape[0])

    #better be random here
    theta = np.zeros((X.shape[1],1))
    print("theta shape " + str(theta.shape))

    for i in range(0,100000):
        h = h_from_x(X,theta)
        l = loss(h,Y)
        #print("loss is " + str(l))
        gradient = grad(X,h,Y)
        theta = learn_theta(theta,gradient)


    return theta

def predict(x,theta):
    h = h_from_x(x,theta)
    class_0 = h-0.0
    class_1 = 1.0 - h
    #print(class_0)
    #print(class_1)
    if class_0 > class_1:
        return "class_1"
    else :
        return "class_0"

    


iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
Y = (iris.target != 0) * 1
Y = np.reshape(Y,(Y.shape[0],1))
#print(X.shape)
#print(Y.shape)
theta = fit(X,Y)
print(theta)
#res = predict(np.array([0.0,1.0]),theta)
for i in range(0,X.shape[0]):
    res = predict(X[i],theta)
    print(str(res) + " " + str(Y[i]))

# res = predict(X[51],theta)
# print(str(res) + " " + str(Y[51]))
#Y = (iris.target != 0) * 1
#print(X.T[0].shape)
#print(X.T[1].shape)
# plt.scatter(X.T[0],X.T[1])
# plt.axis('equal')
# plt.xlim(0, 8)
# plt.ylim(0, 8)
# plt.plot()
# plt.show()