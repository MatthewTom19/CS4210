#-------------------------------------------------------------------------
# AUTHOR: Matthew Tom
# FILENAME: knn.py
# SPECIFICATION: Use of KNN algorithm for binary classification using LOO
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment: 30 mins
#-----------------------------------------------------------*/


#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())


#Loop your data to allow each instance to be your test set
testIndex = 0
correctPredictions = 0
totalPredictions = 0
for i in db:
    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    df_copy_x = df.iloc[:, :-1].copy()
    last_column_name = df_copy_x.columns[-1]
    X = df_copy_x.astype(float)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    last_column = df.iloc[:, -1]
    y_copy = last_column.to_numpy()
    labelEncoder = LabelEncoder()
    y_encoded = labelEncoder.fit_transform(y_copy)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    testSample = X.loc[testIndex].copy()
    test = [testSample]
    X.drop(X.index[testIndex], inplace=True)
    X = X.to_numpy()
    correct_label = y_encoded[testIndex]
    Y = np.delete(y_encoded, testIndex)

    testIndex += 1

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X,Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here

    class_predict = clf.predict(test)[0]
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    #print(f'prediction for sample {testIndex}:', class_predict, f'\nTrue value: {correct_label}\n')

    if class_predict == correct_label:
        correctPredictions += 1
    totalPredictions += 1

#Print the error rate
#--> add your Python code here

print(f'Total Predictions: {totalPredictions}\nWrong Predictions: {totalPredictions-correctPredictions}')
error_rate = (totalPredictions-correctPredictions)/totalPredictions
print('Error rate: ',round(error_rate*100, 2),'%')




