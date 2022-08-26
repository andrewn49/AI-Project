
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import warnings
import pandas as pd

# set data
data = pd.read_csv("dataset.csv")
first = data.values[:, 0:56]
second = data.values[:, 57]
first_train, first_test, second_train, second_test = train_test_split(first, second, test_size=0.1)
print(data)

# knn
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(first_train, second_train)

prediction = knn.predict(first_test)
print("KNN Accuracy:", accuracy_score(second_test, prediction))

crossval = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

scores = cross_val_score(knn, first, second, cv=10)
print("KNN accuracy scores:", scores)
print("KNN accuracy mean:", scores.mean())

warnings.simplefilter('ignore')

# Random Forest
RANDOM_SEED = 0

randfor = RandomForestClassifier(random_state=RANDOM_SEED)
randfor = randfor.fit(first_train, second_train)

prediction = randfor.predict(first_test)
print("Random Forest Accuracy:", accuracy_score(second_test, prediction))

crossval = ShuffleSplit(n_splits=10, test_size=0.1, random_state=RANDOM_SEED)

scores = cross_val_score(randfor, first, second, cv=10)
print("Random Forest accuracy scores:", scores)
print("Random Forest accuracy mean:", scores.mean())


lr = LogisticRegression()

# Decision Tree

dectree = DecisionTreeClassifier(random_state=0, max_depth=2)
dectree = dectree.fit(first_train, second_train)

prediction = dectree.predict(first_test)
print("Decision Tree Accuracy:", accuracy_score(second_test, prediction))

crossval = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

scores = cross_val_score(dectree, first, second, cv=10)
print("Decision Tree accuracy scores:", scores)
print("Decision Tree accuracy mean:", scores.mean())

# Ensemble
stack = StackingCVClassifier(classifiers=[knn, randfor, dectree], meta_classifier=lr, random_state=RANDOM_SEED)

print('10-fold cross validation:')

for clf, label in zip([knn, randfor, dectree, stack],
                      ['KNN', 'Random Forest', 'Decision Tree', 'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, first, second, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    scores = model_selection.cross_val_score(clf, first, second, cv=10, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
