import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import pandas as pd
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


#database cleansing (pandas)
hearts = pd.read_csv('heart_failure_clinical_records_dataset.data',header=0)
names = hearts.columns[:-1].values
heart = hearts.to_numpy()
X = heart[:, :-1]
y = heart[:, 12:]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=1)

#Decision Tree hyperparameters
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth = 6)
tree_clf.fit(x_train,y_train)
pred_tree = tree_clf.predict(x_test)
accuracy_tree = metrics.accuracy_score(y_test, pred_tree)

#Random Forest hyperparameters
rnd_clf = RandomForestClassifier( criterion='entropy',n_estimators=500, max_leaf_nodes=13, n_jobs=-1, random_state=63)
rnd_clf.fit(x_train, y_train)
pred_forest = rnd_clf.predict(x_test)
accuracy_forest = metrics.accuracy_score(y_test, pred_forest)

print('\nDecision Tree accuracy:', accuracy_tree*100)
print('Random Forest accuracy:',accuracy_forest*100)


#user input to test
age = int(input("What's the age of the patient?: "))
anaemia = int(input("Does the patient have anaemia?\n0)No 1)Yes: "))
cpp = int(input("What's the CPP level?: "))
diabetes = int(input("Does the patient have diabetes?\n0)No 1)Yes: "))
ef = int(input("What's the ejection fraction percentage?: "))
hbp = int(input("Does the patient have high blood pressure?\n0)No 1)Yes: "))
platelets = int(input("What's the platelets amount?: "))
sc = float(input("What's the SC level?: "))
ss = int(input("What's the SS level?: "))
sex = int(input("What's the patient's sex?\n0)Female 1)Male: "))
smoke = int(input("Does the patient smoke?\n0)No 1)Yes: "))
time = int(input("How many days ago did the patient suffer the incident?: "))

#Predict
prob_tree = tree_clf.predict_proba([[age,anaemia,cpp,diabetes,ef,hbp,platelets,sc,ss,sex,smoke,time]])
pred_tree =  tree_clf.predict([[age,anaemia,cpp,diabetes,ef,hbp,platelets,sc,ss,sex,smoke,time]])
if pred_tree == 1:
  pred_tree = 'die'
else:
  pred_tree = 'live'
print("\nAccording to the decision tree, the patient is more likely to",pred_tree, "and the probability for the patient to die is:", int(prob_tree[0][1]*100))

prob_forest = tree_clf.predict_proba([[age,anaemia,cpp,diabetes,ef,hbp,platelets,sc,ss,sex,smoke,time]])
pred_forest =  tree_clf.predict([[age,anaemia,cpp,diabetes,ef,hbp,platelets,sc,ss,sex,smoke,time]])
if pred_forest == 1:
  pred_forest = 'Die'
else:
  pred_forest = 'Live'
print("\nAccording to the random forests, the patient is more likely to:",pred_forest, "and the probability for the patient to die is:", int(prob_forest[0][1]*100))
