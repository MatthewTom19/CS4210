#-------------------------------------------------------------------------
# AUTHOR: Matthew Tom
# FILENAME: decision_tree_2.py
# SPECIFICATION: Trains 10 iterations of a decision tree each on 3 different sets of training data of increasing size and calculates the average accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd
import statistics
import copy
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    db_Training = pd.read_csv(ds)

    X = db_Training.iloc[:, :-1]
    Y = db_Training.iloc[:, -1]

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    feature_encoder = OrdinalEncoder ()
    X = feature_encoder.fit_transform(X)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    class_encoder = LabelEncoder()
    Y = class_encoder.fit_transform(Y)

    #Loop your training and test tasks 10 times here
    modelAccuracyResults = []
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here


       testSetDataEncoded = copy.deepcopy(dbTest)

       testEncoder = OrdinalEncoder()
       testSetDataEncoded = testEncoder.fit_transform(testSetDataEncoded)


       #for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here

       correctClassifications = 0
       totalPredictions = 0


       for data in testSetDataEncoded:
           remove_label = data[:-1]
           reshaped_data = remove_label.reshape(1, -1)
           predicted_class = clf.predict(reshaped_data)[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           if predicted_class == data[4]:
               correctClassifications += 1
           totalPredictions += 1
       accuracy = correctClassifications/totalPredictions
       modelAccuracyResults.append(accuracy)


    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

    ten_run_average = statistics.mean(modelAccuracyResults)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    print('Final accuracy when training on', ds, ': ', ten_run_average)




