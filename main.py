import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('Social_Network_Ads.csv')
data.head()

label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

accuracy, classification_rep, confusion_mat

plt.figure(figsize=(20, 10))

plot_tree(clf, filled=True, feature_names=['Gender', 'Age', 'EstimatedSalary'], class_names=['Not Purchased', 'Purchased'])

plt.show()


