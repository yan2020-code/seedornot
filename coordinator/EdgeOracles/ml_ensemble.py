# -*- coding: utf-8 -*-
from ml_engine import *
from online_learning import *
import numpy as np
from collections import deque
import pickle
import os

class EnsembleLearning:
    def __init__(self, save_model_file = None, dataset_path=None, classifier='rf'):
        if save_model_file != None:
            self.ml_model = save_model_file + ".ensemble." + classifier
            self.ol_model = save_model_file + ".ensemble.ol"
            print 'Model is saved here {}'.format(self.ml_model)
            print 'Model is saved here {}'.format(self.ol_model)
        else:
            self.ml_model = None
            self.ol_model = None
        
        # 判断是否使用分类器模式
        self.is_classifier = classifier.endswith('_clf')
        
        # 根据分类器类型选择相应的在线学习模块
        if self.is_classifier:
            # 导入分类器版本的在线学习模块
            from online_learning import OnlineLearningClassifier
            self.online_learning = OnlineLearningClassifier(self.ol_model, dataset_path=dataset_path)
            print 'Using classifier version of OnlineLearning'
        else:
            # 使用传统回归版本
            self.online_learning = OnlineLearningModule(self.ol_model, dataset_path=dataset_path)
        
        self.ml_engine = MLEngine(self.ml_model, classifier=classifier, dataset_path=dataset_path)
        self.classifier = classifier
        
        # 初始化动态加权相关的变量
        self.weight_ml = 0.5  # ML模型的初始权重
        self.weight_ol = 0.5  # Online Learning模型的初始权重
        
        # 权重调整的参数
        self.ADAPTIVE_RATE = 0.1  # 权重调整的学习率
        
        # 尝试加载之前保存的权重
        self.load_weights()


    def predict(self, feature):
        """简化的预测方法"""
        try:
            # 获取预测
            ml_pred = self.ml_engine.predict(feature)
            ol_pred = self.online_learning.predict(feature)
            
            # 计算置信度
            ml_confidence = abs(ml_pred - 0.5) * 2
            ol_confidence = abs(ol_pred - 0.5) * 2
            
            # 基于置信度差异调整权重
            confidence_diff = ml_confidence - ol_confidence
            self.weight_ml += self.ADAPTIVE_RATE * confidence_diff
            self.weight_ol = 1 - self.weight_ml
            
            # 计算加权预测
            prediction = self.weight_ml * ml_pred + self.weight_ol * ol_pred
            
            return prediction
            
        except Exception as e:
            print("预测错误: {}".format(e))
            return 0.5
    
    def update_model(self, features, labels):
        """简化的模型更新方法"""
        try:
            # 更新模型
            self.ml_engine.update_model(features, labels)
            self.online_learning.update_model(features, labels)
                
        except Exception as e:
            print("模型更新错误: {}".format(e))

    def save_model(self):
        self.ml_engine.save_model()
        self.online_learning.save_model()
        
        # 保存当前权重
        if self.ml_model is not None:
            weights_file = self.ml_model + ".weights"
            weights_data = {
                'weight_ml': self.weight_ml,
                'weight_ol': self.weight_ol
            }
            try:
                pickle.dump(weights_data, open(weights_file, 'wb'))
                print "Ensemble weights saved to {}".format(weights_file)
            except Exception as e:
                print "Error saving weights: {}".format(e)
    
    def load_weights(self):
        # 尝试加载之前保存的权重
        if self.ml_model is not None:
            weights_file = self.ml_model + ".weights"
            try:
                if os.path.exists(weights_file):
                    weights_data = pickle.load(open(weights_file, 'rb'))
                    self.weight_ml = weights_data.get('weight_ml', 0.5)
                    self.weight_ol = weights_data.get('weight_ol', 0.5)
                    print "Ensemble weights loaded from {}".format(weights_file)
                    return True
            except Exception as e:
                print "Error loading weights: {}".format(e)
        return False


def testEnsemble():
    print "TEST Ensemble"
    mlengine= EnsembleLearning()
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0,2,1]], [[200]])
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2]], [[143]])
    predicted_value = mlengine.predict([[2,3,2]])
    print "predicted value2: ", predicted_value

def testEnsembleInit():
    print ""
    print("TestInitialDataSet")
    dataset_path = '/home/eric/work/savior/newtcpdump_data.csv'
    mlengine = EnsembleLearning(dataset_path=dataset_path)
    predicted_value = mlengine.predict([[0,2,1,4,5,6,4,5,2,5,5,5]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,6,6,4,2,4,2,5,5,5], [1,3,2,3,1,7,8,4,2,3,4,4]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1,6,7,5,6,4,7,6,6,6]])
    print("predicted value1 : ", predicted_value)

def testAdvancedModels():
    print("")
    print("Testing Advanced Models in Ensemble")
    
    # Test with Bayesian Ridge
    print("Testing Bayesian Ridge")
    mlengine = EnsembleLearning(classifier='bayesian')
    predicted_value = mlengine.predict([[0,2,1,5,4,3,5,5,3,2,2,2,2]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,5,4,3,5,5,3,2,2,2,2]], [[200]])
    
    # Test with Gradient Boosting
    print("Testing Gradient Boosting")
    mlengine = EnsembleLearning(classifier='gbr')
    predicted_value = mlengine.predict([[0,2,1,5,4,3,5,5,3,2,2,2,2]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,5,4,3,5,5,3,2,2,2,2]], [[200]])
    
    # Test with XGBoost if available
    try:
        print("Testing XGBoost")
        mlengine = EnsembleLearning(classifier='xgb')
        predicted_value = mlengine.predict([[0,2,1,5,4,3,5,5,3,2,2,2,2]])
        print("predicted value0 : ", predicted_value)
        mlengine.update_model([[0,2,1,5,4,3,5,5,3,2,2,2,2]], [[200]])
    except:
        print("XGBoost not available for testing")
    
    # Test with LightGBM if available
    try:
        print("Testing LightGBM")
        mlengine = EnsembleLearning(classifier='lgbm')
        predicted_value = mlengine.predict([[0,2,1,5,4,3,5,5,3,2,2,2,2]])
        print("predicted value0 : ", predicted_value)
        mlengine.update_model([[0,2,1,5,4,3,5,5,3,2,2,2,2]], [[200]])
    except:
        print("LightGBM not available for testing")

if __name__ == "__main__":
    testEnsemble()
    testEnsembleInit()
    # Uncomment to test advanced models
    # testAdvancedModels()
