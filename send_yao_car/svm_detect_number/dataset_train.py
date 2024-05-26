import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
'''
function：数据预处理
'''

data = []   # 数据
label = []  # 标签

# 遍历所有数据集
for i in range(1, 9):
    img_list = os.listdir(f'data_set/{i}')
    for each in img_list:
        img = cv2.imread(f'data_set/{i}/{each}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        # 创建数据集和标签
        data.append(img)
        label.append(i)

# 转换二维数组
data = np.array(data).reshape(len(data), -1)
label = np.array(label)

print(data)
print(label)

'''
function: 利用svm进行训练
'''
# 划割数据集
# 定义训练轮数
echo = 5
best_accuracy = 0
for i in range(echo):
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=i)
    # PCA降维 可以降低计算量，提高训练速度，减小模型大小
    transfer = PCA(n_components=300)
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)
    print(X_train.shape)
    print(X_test.shape)

    # 建立svm分类器
    svm = SVC(kernel='linear', probability=True)
    # 使用训练数据来训练SVM分类器
    svm.fit(X_train, Y_train)
    # 使用训练数据来训练SVM分类器
    n_jobs = 4
    # pool = Pool(processes=n_jobs)
    # pool.map(svm.fit(X_train, Y_train)
    #          , range(n_jobs))
    svm.fit(X_train, Y_train)
    # 使用测试数据进行预测
    predict = svm.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(Y_test, predict)
    print(f"Round {i+1}: Accuracy: {accuracy}")

    # 如果当前轮的准确率是所有轮中最高的，则保存当前轮的模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_svm = svm
        # 保存训练好的模型
        joblib.dump(best_svm, 'best_svm_model.joblib')
        joblib.dump(transfer, 'best_pca_model.joblib')
        print(f"Best accuracy achieved: {best_accuracy}")




