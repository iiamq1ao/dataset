# import learn
# from matplotlib import pyplot as plt
# from sklearn import datasets  # 加载数据集
# import numpy as np # 数据处理工具
from datetime import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# import matplotlib.pyplot as plt
# from pylab import xticks, yticks, np
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, mutual_info_classif, RFECV, \
    f_classif
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV  # 随机分训练集
# from sklearn.neural_network import MLPClassifier # 导入神经网络
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import tree
# from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
import sklearn.svm as svm
from sklearn import svm
# from sklearn.metrics import confusion_matrix
# from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import classification_report
# from sklearn.datasets import load_iris
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_score
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# from sklearn import datasets
# from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.svm import LinearSVC
# from scipy.stats import pearsonr
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression
from sklearn import svm
import time

from sklearn.svm import SVC


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
data = read_excel_file('data' + '/intdataall.xlsx')
# print(data.columns)
x=data.iloc[:, :-1]
y=data.iloc[:, -1]
# # 将x和y随机分成训练集和测试集，其中训练集占比0.7，测试集占比0.3
# x, y = make_classification(n_samples=4556, n_features=59, n_classes=6, random_state=1)
# x, y = make_classification(n_samples=4556, n_features=5, random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)


# # # ##################################################SVM#################################################################
# # # # ################################################特征选择前##############################################################
# start_time = time.time()
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# clf = svm.SVC(kernel='linear')
# clf.fit(x_train,y_train)
# y_predict = clf.predict(x_test)
# report = classification_report(y_test, y_predict)
# f1 = f1_score(y_test, y_predict)
# recall = recall_score(y_test, y_predict, average='macro')
# print("svm Accuracy:", accuracy_score(y_test, y_predict),"precision:", precision_score(y_test, y_predict), "  F1值：",f1,"  召回率：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("SVM 模型训练执行时间: ", execution_time, "秒")
# # #recall:recall增加3%


# # # # ##################################################SVM#################################################################
# # # # # ################################################特征选择后##############################################################
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

start_time = time.time()
x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)

# k_values = np.arange(1, x.shape[1] + 1)  # 尝试从1到特征数量的所有可能k值
# best_score = 0
# best_k = 0
# for k in k_values:
#     selector = SelectKBest(f_classif, k=k)
#     X_new = selector.fit_transform(x, y)
#     model = KNeighborsClassifier()  # 使用KNN分类器作为示例
#     scores = cross_val_score(model, X_new, y, cv=2)  # 交叉验证评估模型性能
#     avg_score = np.mean(scores)
#     if avg_score > best_score:
#         best_score = avg_score
#         best_k = k
#         best_feature_indices = selector.get_support(indices=True)  # 获取最佳特征对应的列索引
#
# print("最佳的 k 值:", best_k)
# print("最佳特征的列索引:", best_feature_indices)

# clf = svm.SVC(kernel='linear')
# selector = SelectKBest(f_classif, k=3)
# X_selected = selector.fit_transform(x, y)
# X_train_selected = selector.transform(x_train)
# X_test_selected = selector.transform(x_test)
# clf.fit(X_train_selected, y_train)
# y_predict = clf.predict(X_test_selected)
# accuracy = accuracy_score(y_test, y_predict)
# f1 = f1_score(y_test, y_predict)
# report = classification_report(y_test, y_predict)
# recall = recall_score(y_test, y_predict, average='macro')
# print("svm  accuracy: ", accuracy_score(y_test, y_predict),"precision: ", precision_score(y_test, y_predict),"                     F1：",f1,"  Recall：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("SVM 模型训练执行时间: ", execution_time, "秒")

# # # ##################################################随机森林#################################################################
# # # # ################################################特征选择前##############################################################
# start_time = time.time()
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# rfc = RandomForestClassifier(n_estimators=59, random_state=0)
# rfc.fit(x_train, y_train)
# y_pred = rfc.predict(x_test)
# f1 = f1_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# print("forest Accuracy:", accuracy_score(y_test, y_pred),"precision:", precision_score(y_test, y_pred), "  F1值：",f1,"  召回率：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("forest模型训练执行时间: ", execution_time, "秒")


# ## # # #
# # # ##################################################随机森林#################################################################
# # # # ################################################特征选择后##############################################################
# forest Accuracy: 0.9217264081931237 precision: 0.937125748502994   F1值： 0.9212656364974247   召回率： 0.9059334298118669
# SVM 模型训练执行时间:  0.1295175552368164 秒  3
# forest Accuracy: 0.9246525237746891 precision: 0.9375   F1值： 0.9244314013206162   召回率： 0.9117221418234442
# SVM 模型训练执行时间:  0.1850290298461914 秒  4
#
start_time = time.time()
x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
X_non_negative = np.abs(x)
rfc = RandomForestClassifier(n_estimators=59, random_state=0)

k_values = np.arange(1, x.shape[1] + 1)  # 尝试从1到特征数量的所有可能k值
best_score = 0
best_k = 0
for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(x, y)
    model = KNeighborsClassifier()  # 使用KNN分类器作为示例
    scores = cross_val_score(model, X_new, y, cv=2)  # 交叉验证评估模型性能
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_k = k
        best_feature_indices = selector.get_support(indices=True)  # 获取最佳特征对应的列索引

print("最佳的 k 值:", best_k)
selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(x, y)
X_train_selected = selector.transform(x_train)
X_test_selected = selector.transform(x_test)
rfc.fit(X_train_selected, y_train)
y_pred = rfc.predict(X_test_selected)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("forest Accuracy:", accuracy_score(y_test, y_pred),"precision:", precision_score(y_test, y_pred), "  F1值：",f1,"  召回率：", recall)
end_time = time.time()
execution_time = end_time - start_time
print("SVM 模型训练执行时间: ", execution_time, "秒")


# # ##################################################朴素贝叶斯#################################################################
# # # ################################################特征选择前##############################################################

# start_time = time.time()
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# clfbei = GaussianNB()
# clfbei.fit(x_train, y_train)
# y_pred = clfbei.predict(x_test)
# f1 = f1_score(y_test, y_pred, average='macro')
# recall = recall_score(y_test, y_pred, average='macro')
# print("朴素贝叶斯 accuracy:", accuracy_score(y_test, y_pred),"precision:", precision_score(y_test, y_pred, average='micro'), "  F1值：",f1,"  召回率：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("朴素贝叶斯 模型训练执行时间: ", execution_time, "秒")
# # #

# # ##################################################朴素贝叶斯#################################################################
# # # # # ################################################特征选择后##############################################################
# from sklearn.feature_selection import SelectKBest
# from sklearn.naive_bayes import GaussianNB
# # 假设你有一个特征矩阵 X 和目标变量向量 y
# # 创建朴素贝叶斯分类器对象
# start_time = time.time()
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# naive_bayes = GaussianNB()
# # ######################################################################
# feature_selector = SelectKBest(f_classif, k=3)
# x_selected = feature_selector .fit_transform(x, y)
# x_train_selected = feature_selector .transform(x_train)
# x_test_selected = feature_selector .transform(x_test)
# naive_bayes.fit(x_train_selected, y_train)
# y_pred = naive_bayes.predict(x_test_selected)
# f1 = f1_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='macro')
# print("朴素贝叶斯Accuracy:", accuracy_score(y_test, y_pred),"precision:", precision_score(y_test, y_pred), "  F1值：",f1,"  召回率：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("朴素贝叶斯模型训练执行时间: ", execution_time, "秒")

# # # ##################################################神经网络#################################################################
# # # # # ################################################特征选择前##############################################################

# Neural Network Accuracy: 0.9400146305779078 ANN  Precision: 0.9506172839506173        F1： 0.9375951293759514  Recall： 0.9396379260858576
# # # ANN 模型训练执行时间:  1.1985340118408203 秒
# x, y = make_classification(n_samples=4556, n_features=59, random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# start_time = time.time()
# model = MLPClassifier(hidden_layer_sizes=(3,8), max_iter=2000, random_state=1)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='macro')
# print("Neural Network Accuracy:", accuracy,"ANN  Precision:",precision_score(y_test, y_pred),"       F1：",f1," Recall：",recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("ANN 模型训练执行时间: ", execution_time, "秒")
# # #
# # #
#
# #

# # # ##################################################神经网络#################################################################
# # # # # ################################################特征选择后##############################################################

# model = MLPClassifier(hidden_layer_sizes=(3,8), max_iter=2000, random_state=1)
# model.fit(x_train_selected, y_train)
# y_pred = model.predict(x_test_selected)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='macro')
# print("Neural Network Accuracy:", accuracy,"ANN  Precision:",precision_score(y_test, y_pred),"       F1：",f1," Recall：",recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("ANN 模型训练执行时间: ", execution_time, "秒")

#####################################################################################################
# selector = VarianceThreshold(threshold=0.1)
# x_train_selected = selector.fit_transform(x_train)
# x_test_selected = selector.transform(x_test)
# #####################################################################################################
# model = tf.keras.models.Sequential()  # 创建一个空的Sequential模型
# model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(59,)))
# model.add(tf.keras.layers.Dense(units=64, activation='relu'))
# model.add(tf.keras.layers.Dense(units=6, activation='softmax'))  # 输出层单元数设置为类别数量（6）
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train_selected, y_train, epochs=3, batch_size=32)
# y_pred_selected = model.predict(x_test_selected)
# y_pred_selected = tf.argmax(y_pred_selected, axis=1)
# # 计算评估指标
# accuracy= accuracy_score(y_test, y_pred_selected)
# f1 = f1_score(y_test, y_pred_selected, average='macro')
# recall = recall_score(y_test, y_pred_selected, average='macro')
# precision = precision_score(y_test, y_pred_selected, average='macro')
# print("Neural Network Accuracy:", accuracy)
# print("ANN Precision:", precision)
# print("F1 Score:", f1)
# print("Recall:", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("ANN 模型训练执行时间:", execution_time, "秒")

# # # # ##################################################AdaBoostClassifier#################################################################
# # # # # # ################################################特征选择前##############################################################
# start_time = time.time()
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# # x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=1)
# xAdaBoostClassifier_train, xAdaBoostClassifier_test, yAdaBoostClassifier_train, yAdaBoostClassifier_test = train_test_split(x, y, random_state=1,train_size=0.7)
# modelAdaBoost = AdaBoostClassifier(n_estimators=59, random_state=1)
# modelAdaBoost.fit(xAdaBoostClassifier_train, yAdaBoostClassifier_train)
# AdaBoostaccuracy = modelAdaBoost.score(xAdaBoostClassifier_test, yAdaBoostClassifier_test)
# yAdaBoostClassifier_pred = modelAdaBoost.predict(xAdaBoostClassifier_test)
# f1AdaBoost = f1_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred, average='macro')
# recall = recall_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred)
# # print("AdaBoostClassifier Accuracy:", accuracy , "  F1值：",f1,"  召回率：", recall)
# print("AdaBoostClassifier Accuracy:", AdaBoostaccuracy ,"precision",precision_score(yAdaBoostClassifier_test, yAdaBoostClassifier_pred), "  F1值：",f1AdaBoost,recall)
# #
# end_time = time.time()
# execution_time = end_time - start_time
# print("SVM 模型训练执行时间: ", execution_time, "秒")
# # #
# # # ##################################################AdaBoostClassifier#################################################################
# # # # # ################################################特征选择后##############################################################
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_selection import SelectFromModel
# start_time = time.time()
# #
# x, y = make_classification(n_samples=4556, n_features=59,random_state=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.7)
# # 创建基分类器（决策树）
# base_classifier = DecisionTreeClassifier()
# # base_classifier = RandomForestClassifier()
# # base_classifier = GradientBoostingClassifier()
# # base_classifier = SVC(kernel='rbf', probability=True)
# # base_classifier = GaussianNB()
# #
# # # # 创建AdaBoostClassifier，并使用基分类器作为弱分类器
# adaboost = AdaBoostClassifier(base_estimator=base_classifier)
# # # ################################################(selectbestzuihao)
# selector = SelectKBest(f_classif, k=3)
# x_selected = selector.fit_transform(x, y)
# x_train_selected = selector.transform(x_train)
# x_test_selected = selector.transform(x_test)
# adaboost.fit(x_train_selected, y_train)
# # # 在选择的特征上训练和预测
# adaboost.fit(x_train_selected, y_train)
# y_pred= adaboost.predict(x_test_selected)
# # # # 计算评估指标
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='macro')
# recall= recall_score(y_test, y_pred, average='macro')
# print("AdaBoostClassifier Accuracy:", accuracy_score(y_test, y_pred),"precision:", precision_score(y_test, y_pred, average='micro'), "  F1值：",f1,"  召回率：", recall)
# end_time = time.time()
# execution_time = end_time - start_time
# print("SVM 模型训练执行时间: ", execution_time, "秒")
