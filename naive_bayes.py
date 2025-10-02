#-------------------------------------------------------------------------
# AUTHOR: Matthew Tom
# FILENAME: naive_bayes.py
# SPECIFICATION: using naive bayes to decide if you should play tennis outside
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

X = [row[1:-1] for row in dbTraining]

feature_encoder = OrdinalEncoder ()

X = feature_encoder.fit_transform(X)

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

Y = [row[-1] for row in dbTraining]

class_encoder = LabelEncoder()
Y = class_encoder.fit_transform(Y)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here

clf = MultinomialNB(alpha=1.0)
clf.fit(X, Y)

#Reading the test data using Pandas

df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header as the solution
#--> add your Python code here

column_labels = df.columns.tolist()
finalResult = pd.DataFrame(columns=column_labels)
finalResult['Confidence'] = None

testSet = [row[1:-1] for row in dbTest]
testSetTrimmed = feature_encoder.fit_transform(testSet)

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

count = 1
for sample in testSetTrimmed:
    probability = clf.predict_proba([sample])[0]
    if probability[0] >=0.75:
        tempList = [dbTest[count - 1][:-1]]
        flatten = [item for sublist in tempList for item in sublist]
        flatten.append('no')
        flatten.append(probability)
        finalResult.loc[len(finalResult)] = flatten
    if probability[1] >= 0.75:
        tempList = [dbTest[count-1][:-1]]
        flatten = [item for sublist in tempList for item in sublist]
        flatten.append('yes')
        flatten.append(probability)
        finalResult.loc[len(finalResult)] = flatten

    count +=1
print(finalResult.to_string())