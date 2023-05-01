import sklearn.metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

path = "datasets/_DATA_Original_DONOTALTER/"
dir_list = os.listdir(path)
df_all = []
weights = [2.5, 5, 5, 2.5, 5,
           2.5, 5, 5, 2.5, 5,
           2.5, 5, 5, 5, 5,
           2.5, 5, 2.5, 5, 5,
           2.5, 2.5]
models = [1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 10, 10, 5, 7, 7, 8, 8, 11, 11, 2, 9, 0]

for i in range(len(dir_list)):
    try:
        df = pd.read_csv("datasets/_DATA_Original_DONOTALTER/{}".format(dir_list[i]))
        my_list = ["(AVG)", "(MIN)", "(MAX)", "(RMS)" ]
        needed_cols = []
        for col_name in df.columns:
            if col_name[-8:-3] in my_list or col_name[0] == 'c':
                continue
            needed_cols.append(col_name)
        # print(needed_cols)
        df = df[[x for x in df.columns if x in needed_cols]]
        # for col in df.columns:
        #     print(col)
        df = df.dropna()
        # df = df.dropna(axis=1)
        df['weights'] = weights[i]
        df['model'] = models[i]
        df_all.append(df)
        pd.concat(df_all, axis=0)
    except:
        continue

df = pd.concat(df_all)
df = df.drop("LVDT #6 (194512) []", axis=1)

X = df.drop(columns=['model'])
y = df['model'].values
NUM_OF_NEIGHBORS = 10
max_acc = 0
best_k = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

for i in range(10):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    knn = neighbors.KNeighborsClassifier(n_neighbors=NUM_OF_NEIGHBORS)
    # How do I print out the actual wing number that was predicted (training)? true value (testing)?
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn.predict(X_test)
    score = knn.score(X_test, y_test)
    print("K = {} --> Accuracy: {}".format(NUM_OF_NEIGHBORS, score))
    # print("training set")
    # print(sklearn.metrics.classification_report(y_train, knn.predict(X_train)))
    # print("Test set")
    # print(sklearn.metrics.classification_report(y_test,y_pred))
    if score > max_acc:
        max_acc = score
        best_k = NUM_OF_NEIGHBORS
    NUM_OF_NEIGHBORS += 1
print("Highest accuracy when K = {} with {} accuracy".format(best_k, round(max_acc, 3)))


