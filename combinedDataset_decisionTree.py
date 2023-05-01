import sklearn.metrics
from sklearn import tree
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
            if col_name[-8:-3] in my_list or col_name[0] == 'c' or col_name[0] == 'T':
                continue
            needed_cols.append(col_name)
        df = df[[x for x in df.columns if x in needed_cols]]
        df = df.dropna()
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

dTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)

# dTree.fit(X_train, y_train)
# print(dTree.predict(X_test))
# print(y_test)
# score = dTree.score(X_test, y_test)
# print("Accuracy of decision tree: {}".format(score))

dTree.fit(X_train, y_train)

correct = 0
pred = dTree.predict(X_test)
for i in range(len(y_test)):
    if y_test[i] == pred[i]:
        correct += 1
acc = correct / len(y_test)
print("Accuracy of decision tree: {}".format(acc))



# start value must be less than end value
# manually eyeballing values and their accuracy
start = 68
end = 75

print("{} <-- predicted output".format(pred[start:end]))
print("{} <-- actual output".format(y_test[start:end]))

# print("X_train")
# print(X_train)
# print("y_train")
# print(y_train)




