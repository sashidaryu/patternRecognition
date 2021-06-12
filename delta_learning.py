import numpy as np
from prettytable import PrettyTable
from sklearn import datasets
class Perceptron:
    def __init__(self):
        self.weights=[]

    #activation function
    def activation(self,data):
        #initializing with threshold value
        activation_val=self.weights[0]
        activation_val+=np.dot(self.weights[1:],data)
        val=0
        if activation_val>0: 
          val=1
        elif activation_val==0:
          val=0.5
        else: 
          val=0
        return val

    def fit(self,X,y,lrate, epochs, weights):
        #initializing weight vector
        self.weights=weights
        #no.of iterations to train the neural network
        result = []
        for epoch in range(epochs):
            for index in range(len(X)):
                w_old=self.weights.copy()
                x=X[index]
                xt = np.concatenate(([], x))
                predicted=self.activation(x)
                #check for misclassification
                if (Y[index]==predicted):
                    pass
                else:
                    #calculate the error value
                    error=Y[index]-predicted
                    #updation of threshold
                    self.weights[0]=self.weights[0]+lrate*error
                    #updation of associated self.weights acccording to Perceptron training rule
                    for j in range(len(x)):
                        self.weights[j+1]=self.weights[j+1]+lrate*error*x[j]

                result.append((str((index+1)+(epoch)*len(X)), Y[index], np.round(w_old, 4), predicted, Y[index]-predicted, lrate*(Y[index]-predicted)*xt, np.round(self.weights, 4)))

        pt = PrettyTable(('iteration', 't', 'w_old', 'H(wx)', 't-y', "η(t−y)x", 'w_new'))
        for row in result: pt.add_row(row)

        pt.align['iteration'] = 'c'
        pt.align['t'] = 'l'
        pt.align['w_old'] = 'l'
        pt.align['H(wx)'] = 'l'
        pt.align['t-y'] = 'l'
        pt.align['η(t−y)x'] = 'l'
        pt.align['w_new'] = 'l'

        print(pt)

        print(self.weights)
        counter=0
        for t in range(len(X)):
          x=X[t]
          predicted=self.activation(x)
          if (Y[t]==predicted):
            counter+=1
        print(100*(counter/150))


    #training perceptron for the given data
    def predict(self,x_test):
        predicted=[]
        for i in range(len(x_test)):
            #prediction for test set using obtained weights
            predicted.append(self.activation(x_test.iloc[i]))
        return predicted
    
    def accuracy(self,predicted,original):
        correct=0
        lent=len(predicted)
        for i in range(lent):
            if(predicted[i]==original.iloc[i]):
                correct+=1
        return (correct/lent)*100

    def getweights(self):
        return self.weights


def act(weights,data):
        #initializing with threshold value
        activation_val=weights[0]
        activation_val+=np.dot(weights[1:],data)
        val=0
        if activation_val>0: 
          val=1
        elif activation_val==0:
          val=0.5
        else: 
          val=0
        return val

import pandas as pd
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris['data']
temp=iris['target']
labels=[]
for c in range(len(temp)):
  if temp[c] == 0:
    labels.append(1)
  else:
    labels.append(0)
print(labels)
Y=labels
#splitting test and train data for iris
model=Perceptron()
init_weights=[0.5, -0.5, 3.5, -2.5, 3.5]
#init_weights=[0.5, -2.5, 2.5, 1.5, -0.5]
#init_weights=[0.5, 2.5, -0.5, -3.5, 3.5]
#init_weights=[0.5, 2.5, -1.5, -0.5, -3.5]
l_rate=0.1
epoch=2

counter=0
for t in range(len(X)):
  x=X[t]
  predicted=act(init_weights, x)
  if (Y[t]==predicted):
    counter+=1
print(100*(counter/150))

model.fit(X, Y, l_rate, epoch, init_weights)
