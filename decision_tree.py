#-------------------------------------------------------------------------
# AUTHOR: Matthew Tom
# FILENAME: decision_tree.py
# SPECIFICATION: A decision tree to determine if an individual should be recommended
#                  contact lenses based on their age, spectacle prescription,
#                  astigmatism, and tear production rate
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hour (documentation rabbit-hole)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
#--> add your Python code here

for row in db:
    X.append(row[:-1])
    Y.append(row[-1])

feature_encoder = OrdinalEncoder()
X = feature_encoder.fit_transform(X)

#encode the original categorical training classes into numbers and add to the vector Y.
#--> add your Python code here

class_encoder = LabelEncoder()
Y = class_encoder.fit_transform(Y)
print(class_encoder.classes_)
#fitting the decision tree to the data using entropy as your impurity measure
#--> add your Python code here

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()