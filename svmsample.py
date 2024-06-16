# import learn
# from matplotlib import pyplot as plt
# from sklearn import datasets  # 加载数据集
# import numpy as np # 数据处理工具
from datetime import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, mutual_info_classif, RFECV, \
    f_classif
from sklearn.naive_bayes import GaussianNB
import sklearn.svm as svm
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2

from sklearn import svm
import time

from sklearn.svm import SVC
from sklearn.metrics import precision_score


# # 写一个可以读csv文件的函数
def readfile(filename):
    data = pd.read_csv(filename, header=None)
    return data
# #
# # import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
#
def read_excel_file(filename):
    """
#     读取Excel文件中的数据并返回一个DataFrame对象
#     """
#     # 读取Excel文件并返回DataFrame对象
    df = pd.read_excel(filename)
    return df
#
# #读文件
data = read_excel_file('intdataall.xlsx')
# print(data.columns)
x=data.iloc[:, :-1]
y=data.iloc[:, -1]


# # # ##################################################SVM#################################################################
# # # # ################################################特征选择前##############################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn import svm

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
# report = classification_report(y_test, y_predict, average='macro')
accuracy = clf.score(x_test, y_test)
f1 = f1_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
precision = precision_score(y_test,y_predict, average='macro')
print("svm Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)
# #

# # # # # ##################################################SVM#################################################################
# # # # # # ################################################特征选择后##############################################################
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import time
import numpy as np
start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

clf = svm.SVC(kernel='linear')
selector = SelectKBest(f_classif, k=51)
X_selected = selector.fit_transform(x, y)
X_train_selected = selector.transform(x_train)
X_test_selected = selector.transform(x_test)
clf.fit(X_train_selected, y_train)
y_predict = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
report = classification_report(y_test, y_predict)
recall = recall_score(y_test, y_predict, average='macro')
print("svm  accuracy: ", accuracy, "precision: ", precision_score(y_test, y_predict, average='macro'), "F1：", f1, "Recall：", recall)


# ########################################随机森林#################################################################
# # # # # ################################################特征选择前##############################################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
rfc = RandomForestClassifier(n_estimators=59, random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
accuracy = rfc.score(x_test, y_test)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("forest Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)
# #

# # # ##################################################随机森林#################################################################
# # # # ################################################特征选择后##############################################################
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
X_non_negative = np.abs(x)
rfc = RandomForestClassifier(n_estimators=59, random_state=0)

k_values = np.arange(1, x.shape[1] + 1)  # 尝试从1到特征数量的所有可能k值
best_score = 0
best_k = 0
for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(x, y)
    model = RandomForestClassifier(n_estimators=59, random_state=0)  # 使用随机森林分类器作为示例
    scores = cross_val_score(model, X_new, y, cv=2)  # 交叉验证评估模型性能
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k
        best_feature_indices = selector.get_support(indices=True)  # 获取最佳特征对应的列索引

# print("最佳的 k 值:", best_k)
selector = SelectKBest(f_classif, k=51)
X_selected = selector.fit_transform(x, y)
X_train_selected = selector.transform(x_train)
X_test_selected = selector.transform(x_test)
rfc.fit(X_train_selected, y_train)
y_pred = rfc.predict(X_test_selected)
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print("forest Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)

# #
# # # ##################################################朴素贝叶斯#################################################################
# # # ################################################特征选择前##############################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)


clfbei = GaussianNB()

clfbei.fit(x_train, y_train)
y_pred = clfbei.predict(x_test)
accuracy = clfbei.score(x_test, y_test)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("Naive Bayes Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)


# # ##################################################朴素贝叶斯#################################################################
# # # # # ################################################特征选择后##############################################################
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
naive_bayes = GaussianNB()

k_values = np.arange(1, x.shape[1] + 1)  # 尝试从1到特征数量的所有可能k值
best_score = 0
best_k = 0

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(x, y)
    scores = cross_val_score(naive_bayes, X_new, y, cv=2)  # 交叉验证评估模型性能
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k
        best_feature_indices = selector.get_support(indices=True)  # 获取最佳特征对应的列索引

# print("最佳的 k 值:", best_k)
feature_selector = SelectKBest(f_classif, k=5)
x_selected = feature_selector.fit_transform(x, y)
x_train_selected = feature_selector.transform(x_train)
x_test_selected = feature_selector.transform(x_test)
naive_bayes.fit(x_train_selected, y_train)
y_pred = naive_bayes.predict(x_test_selected)

f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
print("Naive Bayes Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)

# # # # ##################################################神经网络#################################################################
# # # # # # ################################################特征选择前##############################################################
# #
#
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
model = MLPClassifier(hidden_layer_sizes=(3, 8), max_iter=2000, random_state=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print("ANN Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)

# # #
# # #
# # #
#
# # # # ##################################################神经网络#################################################################
# # # # # # ################################################特征选择后##############################################################
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
best_score = 0
best_k = 0

k_values = np.arange(1, x.shape[1] + 1)

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(x, y)
    scores = cross_val_score(GaussianNB(), X_new, y, cv=2)  # 使用朴素贝叶斯模型进行交叉验证
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

# 输出最佳特征数量
# print("最佳的 k 值:", best_k)
# 第一个模型
selector = SelectKBest(f_classif, k=k)
x_train_selected = selector.fit_transform(x_train, y_train)
x_test_selected = selector.transform(x_test)


model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, random_state=1)
model.fit(x_train_selected, y_train)
y_pred = model.predict(x_test_selected)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
print("ANN Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)



# 第二个模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(x_train_selected.shape[1],)))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=6, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_selected, y_train, epochs=10, batch_size=32)  # 增加迭代次数

accuracy, precision, recall, f1 = model.evaluate(x_test_selected, y_test)


print("ANN Accuracy:", accuracy, "precision:", precision, "  F1值：", f1, "  召回率：", recall)







# # # # # ##################################################AdaBoostClassifier#################################################################
# # # # # # # ################################################特征选择前##############################################################

#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import f1_score, recall_score, precision_score
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
# xAdaBoostClassifier_train, xAdaBoostClassifier_test, yAdaBoostClassifier_train, yAdaBoostClassifier_test = train_test_split(x, y, random_state=1, train_size=0.7)
# modelAdaBoost = AdaBoostClassifier(n_estimators=59, random_state=1)
# modelAdaBoost.fit(xAdaBoostClassifier_train, yAdaBoostClassifier_train)
#
# AdaBoostaccuracy = modelAdaBoost.score(xAdaBoostClassifier_test, yAdaBoostClassifier_test)
#
# yAdaBoostClassifier_pred = modelAdaBoost.predict(xAdaBoostClassifier_test)
# f1AdaBoost = f1_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred, average='macro')
#
# precisionAdaBoost = precision_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred, average='macro')
# recallAdaBoost = recall_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred, average='macro')
#
# print("AdaBoostClassifier Accuracy:", AdaBoostaccuracy, "  Precision:", precisionAdaBoost, "  F1-score:", f1AdaBoost, "  Recall:", recallAdaBoost)


#
# # # # #
# # # # # # ##################################################AdaBoostClassifier#################################################################
# # # # # # # # ################################################特征选择后##############################################################
#
# #
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
#
# # 定义基分类器
# base_classifiers = [RandomForestClassifier(), GradientBoostingClassifier(), SVC(kernel='rbf', probability=True),
#                     GaussianNB(), DecisionTreeClassifier()]
#
# best_score = 0
# best_k = 0
# best_classifier = None
#
# k_values = np.arange(1, x.shape[1] + 1)
#
# for classifier in base_classifiers:
#     for k in k_values:
#         selector = SelectKBest(f_classif, k=k)
#         X_new = selector.fit_transform(x, y)
#         scores = cross_val_score(AdaBoostClassifier(n_estimators=59), X_new, y, cv=2)  # 使用AdaBoost进行交叉验证
#         avg_score = np.mean(scores)
#         if avg_score > best_score:
#             best_score = avg_score
#             best_k = k
#             best_classifier = classifier
#
# adaboost = AdaBoostClassifier(n_estimators=59)
#
# selector = SelectKBest(f_classif, k=best_k)
# x_train_selected = selector.fit_transform(x_train, y_train)
# x_test_selected = selector.transform(x_test)
#
# adaboost.fit(x_train_selected, y_train)
# y_pred = adaboost.predict(x_test_selected)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# precision = precision_score(y_test, y_pred, average='micro')
#
# print("AdaBoostClassifier Accuracy:", accuracy, "Precision:", precision, "  F1值：", f1, "  召回率：", recall)
