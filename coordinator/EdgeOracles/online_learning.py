"""
Online learning module
By: Boyu Wang (boywang@cs.stonybrook.edu)
    Yaohui Chen (yaohway@gmail.com)
Created Date: 2 Jun 2019
Last Modified Date: 17 June 2019
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model.ridge import Ridge
from tempfile import TemporaryFile
import os

# utility functions
# append 1 to the features, act as a bias term
def utl_add_bias(features):
    features = np.array(features)
    features = np.concatenate([features, np.ones([features.shape[0], 1])], axis=1)
    return features

# for features and scores, it would be better if you could normalize them first before fitting into the model
class OnlineLearningModule:
    def __init__(self, save_model_file = None, dataset_path=None):
        #load Cn_inv and W from file
        try:
            self.save_model_file = save_model_file
            self.load_model()
            self.is_init = True
        except Exception:
            self.is_init = False

        if dataset_path is not None:
            print "reading data from: ",dataset_path
            dataset = pd.read_csv(dataset_path, delimiter=',')
            print "dataset shape: ",dataset.shape
            labels = dataset.label
            print "labels shape: ",labels.shape
            dataset.drop('window', axis=1, inplace=True)
            dataset.drop('label', axis=1, inplace=True)
            dataset.drop('id', axis=1, inplace=True)
            features = dataset
            print "features shape: ",features.shape
            # train model with the initial data if no model is provided
            if save_model_file is not None:
                self.update_model(features, labels)


    def load_model(self):
        f = self.save_model_file
        model = pickle.load(open(f,'rb'))
        print("Loading model from {0}".format(f))
        self.W = model['W']
        self.Cn_inv = model['Cn_inv']

    def save_model(self):
        print "[Online learning] Saving model"	
        if not self.save_model_file is None:
            f = self.save_model_file
            print("Saving model to {0}".format(f))
            model = {'W':self.W, 'Cn_inv':self.Cn_inv}
            pickle.dump(model, open(f,'wb'))

    # features: numpy array of size [n, fea_dim]
    # scores: numpy array of size [n]
    # alpha is the weight for l2 regularization, can be tuned
    def first_update(self, features, scores, alpha):
        self.alpha = alpha

        # conversion from list to numpy array
        features = np.array(features)
        scores = np.array(scores)

        features = utl_add_bias(features)

        init_clf = Ridge(alpha=self.alpha, fit_intercept=False, random_state=0)
        init_clf.fit(features, scores)

        # W: numpy matrix of size [fea_dim+1, 1]
        init_W = np.matrix(init_clf.coef_).T
        features = np.matrix(features)
        Cn = features.T * features + self.alpha * np.eye(features.shape[1])
        Cn_inv = np.linalg.inv(Cn)
        XiYi = np.expand_dims(np.sum(np.array(features) * np.expand_dims(scores, 1), axis=0), 1) # [d, 1]

        # self.Cn = Cn
        self.Cn_inv = Cn_inv
        self.W = init_W
        print "Weight: ", self.W.shape
        print "Cn_inv: ", self.Cn_inv.shape
        # print(np.sum(np.abs(Cn_inv * XiYi - self.W)))
        self.is_init = True

    # update the classifier with new batch of data
    # features: numpy array of size [n, fea_dim]
    # scores: numpy array of size [n]
    def update_model(self, features, scores, alpha=1.0):
        if not self.is_init:
            self.first_update(features, scores, alpha)
            # print "after first update W: ", self.W
            return
        # print "prev W: ", self.W

        # conversion from list to numpy array
        features = np.array(features)
        scores = np.array(scores)

        features = utl_add_bias(features)
        [n, d] = features.shape
        features = np.matrix(features) # [n, d]
        scores = np.matrix(scores).T # [n, 1]
        print features.shape
        print self.Cn_inv.shape
        denom = features * self.Cn_inv * features.T + np.eye(n)
        # denom = np.matmul(np.matmul(features, self.Cn_inv), features.transpose()) + np.eye(n)
        denom = np.linalg.inv(denom)
        # update Cn_inv
        Cn_inv = self.Cn_inv - self.Cn_inv * features.T * denom *features * self.Cn_inv
        self.W = self.W + Cn_inv * (features.T * scores - features.T * features * self.W)
        # Cn_inv = self.Cn_inv - np.matmul(np.matmul(np.matmul(np.matmul(self.Cn_inv, features.transpose()), denom), features), self.Cn_inv)
        # self.W = self.W + np.matmul(Cn_inv, np.matmul(features.transpose(), scores) - np.matmul(np.matmul(Xk, Xk.transpose()), last_coef))
        # print "after W: ", self.W

    # get the predicted scores for the input features
    # features: numpy array of size [n, fea_dim]
    # return: scores: [n]
    def predict(self, features):
        try:
            # conversion from list to numpy array
            features = np.array(features)

            # append 1 to the features
            features = utl_add_bias(features)
            scores = (features * self.W).A1
        except Exception:
            # conversion from list to numpy array
            features = np.array(features)
            features = utl_add_bias(features)
            # print len(features), len(features[0])
            # print features
            self.W = np.matrix(np.ones([len(features[0]), 1]))
            # print "score: ", features * self.W
            scores = (features * self.W).A1

        return scores[0]

"""
Online learning classifier module - 支持分类的在线学习模块
"""

from sklearn.linear_model import SGDClassifier

class OnlineLearningClassifier:
    """支持分类的在线学习模块，使用SGDClassifier实现"""
    
    def __init__(self, save_model_file=None, dataset_path=None):
        # 分类器参数
        self.clf = SGDClassifier(
            loss='log_loss',  # 逻辑回归损失函数
            alpha=0.0001,     # 正则化强度
            max_iter=1000,    # 最大迭代次数
            tol=1e-3,         # 停止条件容忍度
            class_weight='balanced',  # 自动处理类别不平衡
            random_state=42,  # 随机种子，确保结果可复现
            warm_start=True,  # 允许增量训练
            n_jobs=-1         # 使用所有可用CPU
        )
        
        self.save_model_file = save_model_file
        self.is_init = False
        
        # 尝试加载已有模型
        if save_model_file is not None:
            try:
                self.load_model()
                self.is_init = True
                print("成功加载分类器模型: {}".format(save_model_file))
            except Exception as e:
                print("无法加载分类器模型: {}".format(e))
        
        # 如果提供了数据集路径，则使用初始数据集训练
        if dataset_path is not None:
            print("从数据集初始化分类器: {}".format(dataset_path))
            try:
                dataset = pd.read_csv(dataset_path, delimiter=',')
                labels = dataset.label
                
                # 转换为二分类标签: 大于0的为1，其余为0
                binary_labels = (labels > 0).astype(int)
                
                # 打印正例比例
                positive_ratio = np.sum(binary_labels) / float(len(binary_labels))
                print("正例比例: {:.2f}%".format(positive_ratio * 100))
                
                # 处理特征
                dataset.drop('window', axis=1, inplace=True, errors='ignore')
                dataset.drop('label', axis=1, inplace=True, errors='ignore')
                dataset.drop('id', axis=1, inplace=True, errors='ignore')
                features = dataset
                
                # 使用初始数据训练模型
                self.update_model(features, labels)
                print("初始分类器训练完成，使用{}个样本".format(len(binary_labels)))
            except Exception as e:
                print("初始化分类器失败: {}".format(e))
    
    def load_model(self):
        """从文件加载模型"""
        if self.save_model_file and os.path.exists(self.save_model_file + '.clf'):
            with open(self.save_model_file + '.clf', 'rb') as f:
                self.clf = pickle.load(f)
            print("从 {} 加载分类器模型".format(self.save_model_file + '.clf'))
    
    def save_model(self):
        """保存模型到文件"""
        if self.save_model_file:
            with open(self.save_model_file + '.clf', 'wb') as f:
                pickle.dump(self.clf, f)
            print("分类器模型保存到 {}".format(self.save_model_file + '.clf'))
    
    def update_model(self, features, labels):
        """使用新数据更新模型"""
        if len(features) == 0 or len(labels) == 0:
            print("警告: 空数据集，跳过更新")
            return
            
        # 高效转换为numpy数组
        if isinstance(features[0], np.ndarray):
            X = features
        else:
            X = np.array(features)
        y = np.array(labels)
        
        # 转换为二分类标签
        binary_y = (y > 0).astype(int)
        
        # 打印正例比例
        if len(binary_y) > 0:
            positive_ratio = np.sum(binary_y) / float(len(binary_y))
            print("更新数据正例比例: {:.2f}%".format(positive_ratio * 100))
        
        try:
            if not self.is_init:
                # 首次训练，需要指定所有可能的类别
                self.clf.partial_fit(X, binary_y, classes=np.array([0, 1]))
                self.is_init = True
            else:
                # 增量训练
                self.clf.partial_fit(X, binary_y)
            print("分类器更新成功，使用{}个样本".format(len(binary_y)))
        except Exception as e:
            print("分类器更新失败: {}".format(e))
    
    def predict(self, features):
        """预测输入特征的概率分数"""
        try:
            # 高效转换为numpy数组
            if isinstance(features[0], np.ndarray):
                X = features[0].reshape(1, -1)
            elif isinstance(features[0], list):
                X = np.array(features[0]).reshape(1, -1)
            else:
                X = np.array(features).reshape(1, -1)
            
            if not self.is_init:
                print("警告: 模型未初始化，返回随机权重")
                # 返回0.5作为默认概率
                return 0.5
            
            # 获取正类概率
            proba = self.clf.predict_proba(X)
            return proba[0][1]  # 返回正类(类别1)的概率
            
        except Exception as e:
            print("分类器预测失败: {}".format(e))
            # 初始化预测器失败时返回默认值
            return 0.5

# usage
def test():
    print ""
    print("Test1")
    d = 5
    n = 101
    alpha = 1.0
    features = np.random.rand(n, d)
    scores = np.random.rand(n)
    print "labels shape: ",scores.shape
    print "features shape: ",features.shape

    # initialize the model
    print('initialize the model')
    model = OnlineLearningModule()
    model.update_model(features, scores, alpha)

    # test the model
    pred_scores = model.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('average prediction error on the first batch:{}'.format(err1))

    # update the model with new data
    features2 = np.random.rand(n, d)
    scores2 = np.random.rand(n)
    print('update model with new data')
    model.update_model(features2, scores2)

    pred_scores2 = model.predict(features2)
    err2 = np.linalg.norm(pred_scores2-scores2) / n
    print('average prediction error on the second batch:{}'.format(err2))
    pred_scores = model.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('after update, average prediction error on the first batch:{}'.format(err1))

    # compare online learning with offline learning, make sure they have same result
    print('compare online learning with offline learning')
    features_all = np.concatenate([features, features2], axis=0)
    scores_all = np.concatenate([scores, scores2])
    model2 = OnlineLearningModule()
    model2.update_model(features_all, scores_all, alpha)
    # test the model
    pred_scores = model2.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('offline learning, average prediction error on the first batch:{}'.format(err1))
    pred_scores2 = model2.predict(features2)
    err2 = np.linalg.norm(pred_scores2-scores2) / n
    print('offline learning, average prediction error on the second batch:{}'.format(err2))

def test2():
    print ""
    print("Test2")
    mlengine= OnlineLearningModule()
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1], [2,3,1]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value1 : ", predicted_value)
    mlengine.update_model([[1,3,2]], [24])
    predicted_value = mlengine.predict([[2,3,2]])
    print("predicted value2: ", predicted_value)

def test3():
    print ""
    print("Test3")
    outf = "/tmp/.testdump"
    mlengine = OnlineLearningModule(outf)
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1], [2,3,1]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value1 : ", predicted_value)
    mlengine.update_model([[1,3,2]], [24])
    mlengine.save_model()

    mlengine2 = OnlineLearningModule(outf)
    predicted_value = mlengine.predict([[2,3,2]])
    print("predicted value2: ", predicted_value)

def test4():
    print ""
    print("TestInitialDataSet")
    dataset_path = '/home/eric/work/savior/newtcpdump_data.csv'
    mlengine = OnlineLearningModule(dataset_path=dataset_path)
    predicted_value = mlengine.predict([[0,2,1,4,5,5,6,2,1,6,6,6,6]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,6,6,8,9,0,4,2,2,2,5], [1,3,2,3,1,8,5,6,4,3,6,6,6]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1,6,7,2,5,3,5,6,5,5,5]])
    print("predicted value1 : ", predicted_value)

def test_online_classifier():
    print ""
    print("Test Online Classifier")
    clf = OnlineLearningClassifier()
    
    # 测试预测
    predicted_prob = clf.predict([[0,2,1,5,4,3]])
    print("Initial prediction probability: ", predicted_prob)
    
    # 测试更新 - 正例和负例
    clf.update_model([[0,2,1,5,4,3], [2,3,1,1,2,3]], [5, 0])
    predicted_prob = clf.predict([[0,2,1,5,4,3]])
    print("Prediction after update: ", predicted_prob)
    
    # 再次更新
    clf.update_model([[1,3,2,4,5,6]], [10])
    predicted_prob = clf.predict([[2,3,2,5,6,7]])
    print("Final prediction: ", predicted_prob)

if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()
    
    # 添加分类器测试
    test_online_classifier()

