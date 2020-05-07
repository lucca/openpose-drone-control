from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import numpy as np
import csv
from joblib import dump, load

poses = [
         'back', 'down', 'forward',
         'left', 'neutral', 'right',
         'turnleft', 'turnright', 'up'
        ]

X = []
y = []

# Load data from csv set
for p in poses:
    with open(f'data/{p}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            X.append(np.array(row, dtype=np.single))
            y.append(p)

X = np.array(X)
y = np.array(y)

mlp = MLPClassifier()

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True)
print("Score over 5 folds:")
for train_index, test_index in kf.split(X):
    mlp.fit(X[train_index], y[train_index])
    print(mlp.score(X[test_index], y[test_index]))

mlp.fit(X, y)

dump(mlp, 'model.joblib')