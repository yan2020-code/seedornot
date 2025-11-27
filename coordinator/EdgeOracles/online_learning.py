import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model.ridge import Ridge
from sklearn import exceptions as sklearn_exceptions
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
            dataset = pd.read_csv(dataset_path, delimiter=',')
            labels = dataset.label
            dataset.drop('window', axis=1, inplace=True)
            dataset.drop('label', axis=1, inplace=True)
            dataset.drop('id', axis=1, inplace=True)
            features = dataset
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
            if not self.is_init or not hasattr(self, 'W') or self.W is None:
                return self._heuristic_predict(features)
            
            features = np.array(features)

            features = utl_add_bias(features)
            scores = (features * self.W).A1
        except Exception:
            return self._heuristic_predict(features)

        return scores[0]
    
    def _heuristic_predict(self, features):
        """Heuristic prediction method when regressor model is not initialized"""
        try:
            features = np.array(features)
            if isinstance(features[0], (list, np.ndarray)):
                features_array = np.asarray(features[0])
            else:
                features_array = np.array(features)
            
            return sum(features_array)
        except Exception as e2:
            return 0.0

from sklearn.linear_model import SGDClassifier

class OnlineLearningClassifier:
    """Online learning classifier module using SGDClassifier"""
    
    def __init__(self, save_model_file=None, dataset_path=None):
        self.clf = SGDClassifier(
            loss='log',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True,
            n_jobs=-1
        )
        
        self.save_model_file = save_model_file
        self.is_init = False
        
        self.pending_features = []
        self.pending_labels = []
        
        if save_model_file is not None:
            try:
                self.load_model()
                self.is_init = True
            except Exception as e:
                msg = "Failed to load classifier model: {}".format(e)
                print(msg)
        
        if dataset_path is not None:
            try:
                dataset = pd.read_csv(dataset_path, delimiter=',')
                labels = dataset.label
                
                dataset.drop('window', axis=1, inplace=True, errors='ignore')
                dataset.drop('label', axis=1, inplace=True, errors='ignore')
                dataset.drop('id', axis=1, inplace=True, errors='ignore')
                features = dataset
                
                self.update_model(features, labels)
            except Exception as e:
                 msg = "Failed to initialize classifier: {}".format(e)
                 print(msg)
                 import traceback
                 traceback.print_exc()
    
    def load_model(self):
        """Load model from file"""
        if self.save_model_file and os.path.exists(self.save_model_file + '.clf'):
            with open(self.save_model_file + '.clf', 'rb') as f:
                self.clf = pickle.load(f)
    
    def save_model(self):
        """Save model to file"""
        try:
            if self.save_model_file:
                with open(self.save_model_file + '.clf', 'wb') as f:
                    pickle.dump(self.clf, f)
        except Exception as e:
            print "Warning: Failed to save OnlineLearningClassifier model: {}".format(e)
    
    def update_model(self, features, labels):
        """Update model with new data"""
        try:
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            elif isinstance(features, pd.Series):
                features_array = features.values.reshape(1, -1)
            else:
                features_array = features
            
            if isinstance(labels, pd.Series):
                labels_array = labels.values
            else:
                labels_array = labels
            
            features_len = len(features_array) if hasattr(features_array, '__len__') else 0
            labels_len = len(labels_array) if hasattr(labels_array, '__len__') else 0
            
            if features_len == 0 or labels_len == 0:
                msg = "WARNING: Empty dataset, skip update (features_len={}, labels_len={})".format(
                    features_len, labels_len)
                print(msg)
                return
            
            if not self.is_init:
                if isinstance(features_array, np.ndarray):
                    if features_array.ndim == 1:
                        features_array = features_array.reshape(1, -1)
                else:
                    features_array = np.array(features_array)
                    if features_array.ndim == 1:
                        features_array = features_array.reshape(1, -1)
                
                self.pending_features.append(features_array)
                self.pending_labels.append(labels_array)
                
                if len(self.pending_features) == 1:
                    all_features = self.pending_features[0]
                    all_labels = self.pending_labels[0]
                else:
                    all_features = np.vstack(self.pending_features)
                    all_labels = np.concatenate(self.pending_labels)
                
                all_labels = np.atleast_1d(np.array(all_labels, dtype=float))
                binary_y = (all_labels > 0).astype(int)
                unique_classes = np.unique(binary_y)
                
                if len(unique_classes) == 2:
                    try:
                        self.first_update(all_features, all_labels)
                        self.pending_features = []
                        self.pending_labels = []
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise
                return
            
            features = np.array(features_array)
            labels = np.atleast_1d(np.array(labels_array, dtype=float))
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            binary_y = (labels > 0).astype(int)
            
            try:
                self.clf.partial_fit(features, binary_y, classes=np.array([0, 1]))
            except (ValueError, sklearn_exceptions.NotFittedError) as e:
                if "classes must be passed" in str(e) or isinstance(e, sklearn_exceptions.NotFittedError):
                    unique_classes = np.unique(binary_y)
                    if len(unique_classes) == 2:
                        self.is_init = False
                        self.pending_features = [features]
                        self.pending_labels = [labels]
                        self.first_update(features, labels)
                else:
                    raise
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def first_update(self, features, labels):
        """First training, similar to regression model's first_update"""
        try:
            features = np.array(features)
            labels = np.atleast_1d(np.array(labels, dtype=float))
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            binary_y = (labels > 0).astype(int)
            unique_classes = np.unique(binary_y)
            
            if len(unique_classes) != 2:
                msg = "ERROR: first_update called with only {} class(es), expected 2".format(len(unique_classes))
                print(msg)
                raise ValueError("first_update requires both classes to be present")
            
            self.clf.partial_fit(features, binary_y, classes=np.array([0, 1]))
            self.is_init = True
        except Exception as e:
            msg = "Classifier initialization failed: {}".format(e)
            print(msg)
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, features):
        """Predict probability score for input features"""
        try:
            if not self.is_init:
                if len(self.pending_features) > 0:
                    if len(self.pending_features) == 1:
                        all_features = self.pending_features[0]
                        all_labels = self.pending_labels[0]
                    else:
                        all_features = np.vstack(self.pending_features)
                        all_labels = np.concatenate(self.pending_labels)
                    
                    binary_y = (all_labels > 0).astype(int)
                    unique_classes = np.unique(binary_y)
                    
                    if len(unique_classes) == 2:
                        try:
                            self.first_update(all_features, all_labels)
                            self.pending_features = []
                            self.pending_labels = []
                        except Exception as e:
                            return self._heuristic_predict(features)
                    else:
                        return self._heuristic_predict(features)
                else:
                    return self._heuristic_predict(features)
            
            features = np.array(features)
            
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            if not hasattr(self.clf, 'classes_') or self.clf.classes_ is None:
                return self._heuristic_predict(features)
            
            proba = self.clf.predict_proba(features)
            prob = proba[0][1]
            return prob
        except (sklearn_exceptions.NotFittedError, AttributeError, ValueError) as e:
            if isinstance(e, sklearn_exceptions.NotFittedError):
                self.is_init = False
            return self._heuristic_predict(features)
        except Exception as e:
            return self._heuristic_predict(features)
    
    def _heuristic_predict(self, features):
        """Heuristic prediction method when model is not initialized"""
        try:
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            features_with_bias = utl_add_bias(features)
            default_weights = np.matrix(np.ones([features_with_bias.shape[1], 1]))
            features_with_bias = np.matrix(features_with_bias)
            linear_score = (features_with_bias * default_weights).A1[0]
            
            prob = 1.0 / (1.0 + np.exp(-linear_score / 10.0))
            return float(prob)
        except Exception as e2:
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
    
    predicted_prob = clf.predict([[0,2,1,5,4,3]])
    print("Initial prediction probability: ", predicted_prob)
    
    clf.update_model([[0,2,1,5,4,3], [2,3,1,1,2,3]], [5, 0])
    predicted_prob = clf.predict([[0,2,1,5,4,3]])
    print("Prediction after update: ", predicted_prob)
    
    clf.update_model([[1,3,2,4,5,6]], [10])
    predicted_prob = clf.predict([[2,3,2,5,6,7]])
    print("Final prediction: ", predicted_prob)

if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()
    
    test_online_classifier()
