#-------------------------------------------------------------------------
# AUTHOR: Matthew Tom
# FILENAME: perceptron.py
# SPECIFICATION: Train a perceptron and a multilayer perceptron with varying learning rates and shuffle.
#                They will attempt to identify digits based on ascii-encoded training samples of digits
# FOR: CS 4210- Assignment #3
# TIME SPENT: 14 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

best_acc_P = 0
best_acc_MLP = 0

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r
        clfP = Perceptron(max_iter=1000, eta0=1.0, shuffle=shuffle)
        clfMLP = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(25,),
                               shuffle=shuffle, max_iter=1000)
        clfP.fit(X_training, y_training)
        clfMLP.fit(X_training, y_training)

        P_correct = 0
        MLP_correct = 0
        sample_count = 0

        for (x_testSample, y_testSample) in zip(X_test, y_test):
            P_predict = clfP.predict([x_testSample])
            truth = [y_testSample]
            #print(predict, truth)
            if P_predict == truth:
                P_correct += 1
            MLP_predict = clfMLP.predict([x_testSample])
            if MLP_predict == truth:
                MLP_correct += 1
            sample_count += 1
        P_acc = P_correct/sample_count
        MLP_acc = MLP_correct/sample_count

        if P_acc > best_acc_P:
            print("Highest Perceptron accuracy = ", P_acc, "Parameters: learning rate=", learning_rate, "Shuffle=", shuffle)
            best_acc_P = P_acc
        if MLP_acc > best_acc_MLP:
            print("Highest MLP accuracy = ", MLP_acc, "Parameters: learning rate=", learning_rate, "Shuffle=", shuffle)
            best_acc_MLP = MLP_acc


        #iterates over both algorithms
        #-->add your Python code here

        #for : #iterates over the algorithms

            #Create a Neural Network classifier
                #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
                #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Python code here

            #Fit the Neural Network to the training data


            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here











