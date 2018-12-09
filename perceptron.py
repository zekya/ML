# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:45:31 2018

@author: fu_ji
"""
import numpy as np
import pandas as pd

class perceptron:
    def __init__(self,n_itr):
        self.n_itr=n_itr
        
    def predict(self,row,weight):
        return 1 if np.dot(row,weight)>=0 else -1
    
    def pred(self,sample):
        sample=np.concatenate([sample,np.ones((1,))])
        return 1 if np.dot(sample,self.weight)>=0 else -1
    
    def fit(self, X,Y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Y : {array-like}, shape = [n_samples]\
        """
        one = np.ones((X.shape[0],1))
        X=np.concatenate([X,one],axis=1)
        
        weight = np.zeros(len(X[0]))
        self.weight=weight
        
        for j in range(self.n_itr):
            error= False
            for i in range(len(X)):
                if self.predict(X[i],weight)!=Y[i]:
                    weight=weight+np.dot(Y[i],X[i])
                    error = True
                    break
            if not error:
                break
        self.weight = weight
        return self
if __name__ =='__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    X=df.iloc[:,0:2]
    Y= np.array(list(map(lambda x: 1 if x=='Iris-setosa' else -1, df[4])))
    model = perceptron(10000)
    model.fit(X,Y)
    count=0
    for i in range(len(X)):
        if model.pred(X.iloc[i])==Y[i]:
            count+=1
    accuracy = count/len(X)
    print('accuracy: %f' %accuracy)   
    
        