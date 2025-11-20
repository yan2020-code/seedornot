# -*- coding: utf-8 -*-
from ml_engine import *
from online_learning import *
import pickle
import os

class EnsembleLearning:
    def __init__(self, save_model_file = None, dataset_path=None, classifier='rf'):
        if save_model_file != None:
            self.ml_model = save_model_file + ".ensemble." + classifier
            self.ol_model = save_model_file + ".ensemble.ol"
            #print 'Model is saved here {}'.format(self.ml_model)
            #print 'Model is saved here {}'.format(self.ol_model)
        else:
            self.ml_model = None
            self.ol_model = None
        
        if classifier.endswith('_clf'):
            self.online_learning = OnlineLearningClassifier(self.ol_model, dataset_path=dataset_path)
        else:
            self.online_learning = OnlineLearningModule(self.ol_model, dataset_path=dataset_path)
        
        self.ml_engine = MLEngine(self.ml_model, classifier=classifier, dataset_path=dataset_path)
        self.classifier = classifier
        
        self.weight_ml = 0.5
        self.weight_ol = 0.5
        self.ADAPTIVE_RATE = 0.1
        self.load_weights()


    def predict(self, feature):
        try:
            ml_pred = self.ml_engine.predict(feature)
            ol_pred = self.online_learning.predict(feature)
            
            ml_confidence = abs(ml_pred - 0.5) * 2
            ol_confidence = abs(ol_pred - 0.5) * 2
            
            confidence_diff = ml_confidence - ol_confidence
            self.weight_ml += self.ADAPTIVE_RATE * confidence_diff
            self.weight_ol = 1 - self.weight_ml
            
            prediction = self.weight_ml * ml_pred + self.weight_ol * ol_pred
            
            return prediction
            
        except Exception as e:
            print("Prediction error: {}".format(e))
            return 0.5
    
    def update_model(self, features, labels):
        try:
            self.ml_engine.update_model(features, labels)
            self.online_learning.update_model(features, labels)
        except Exception as e:
            print("Model update error: {}".format(e))

    def save_model(self):
        self.ml_engine.save_model()
        self.online_learning.save_model()
        
        if self.ml_model is not None:
            weights_file = self.ml_model + ".weights"
            weights_data = {
                'weight_ml': self.weight_ml,
                'weight_ol': self.weight_ol
            }
            try:
                pickle.dump(weights_data, open(weights_file, 'wb'))
            except Exception as e:
                pass
        
    
    def load_weights(self):
        if self.ml_model is not None:
            weights_file = self.ml_model + ".weights"
            try:
                if os.path.exists(weights_file):
                    weights_data = pickle.load(open(weights_file, 'rb'))
                    self.weight_ml = weights_data.get('weight_ml', 0.5)
                    self.weight_ol = weights_data.get('weight_ol', 0.5)
                    return True
            except Exception as e:
                pass
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

if __name__ == "__main__":
    testEnsemble()
    testEnsembleInit()
