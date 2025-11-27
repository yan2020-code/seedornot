from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.svm import SVR
import sklearn
import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score

try:
    import xgboost as xgb
except ImportError:
    #print "XGBoost not installed, xgb classifier will not be available"
try:
    import lightgbm as lgb
except ImportError:
    #print "LightGBM not installed, lgbm classifier will not be available"


def _add_bias(features):
    """Add bias term to feature vector"""
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = np.concatenate([features, np.ones([features.shape[0], 1])], axis=1)
    return features


class MLEngine:

    def __init__(self, model_dir=None, classifier='rf', dataset_path=None,
                 columns=list()):
        model_file_name = 'reachability_model.pkl'
        self.classifier = classifier
        
        self.is_classifier = classifier.endswith('_clf')
        
        if len(columns) == 0:
            self.columns = ['reachable label', 'path length',
            'undiscovered neighbours', 'new cov', 'size', 'cmp',
            'indcall', 'extcall', 'reached labels', 'queue size']
        else:
            self.columns = columns
        self.best_model_params = {}

        self.features = pd.DataFrame(columns=self.columns)
        self.labels = []
        
        # Calculate class weight for imbalanced data
        self.class_weight = None
        self.scale_pos_weight = 20.0
        
        if self.classifier == 'rf':
            if dataset_path is None:
                self.clf = RandomForestRegressor(n_estimators=10, max_depth=4)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'rf_clf':
            if dataset_path is None:
                self.clf = RandomForestClassifier(n_estimators=50, max_depth=4, class_weight='balanced')
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'svr':
            if dataset_path is None:
                self.clf = SVR(kernel='linear', C=1.0)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'svr_rbf':
            if dataset_path is None:
                self.clf = SVR(kernel='rbf', gamma='scale', C=1.0)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'bayesian':
            if dataset_path is None:
                self.clf = BayesianRidge(n_iter=300, alpha_1=1e-6, alpha_2=1e-6, 
                                         lambda_1=1e-6, lambda_2=1e-6)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'gbr':
            if dataset_path is None:
                self.clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                                   max_depth=3, min_samples_split=2,
                                                   min_samples_leaf=1, subsample=0.8)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'xgb':
            try:
                if dataset_path is None:
                    self.clf = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, 
                                              max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                              scale_pos_weight=self.scale_pos_weight,
                                              reg_alpha=0.1, reg_lambda=1.0,
                                              min_child_weight=3)
                    self.dataset_path = ''
                else:
                    self.dataset_path = dataset_path
                    print 'Initializing model from: ', self.dataset_path
                    self.build_model()
            except NameError:
                print 'XGBoost is not installed. Install with: pip install xgboost'
                exit(1)
        elif self.classifier == 'xgb_clf':
            try:
                if dataset_path is None:
                    self.clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, 
                                               max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                               scale_pos_weight=self.scale_pos_weight,
                                               reg_alpha=0.1, reg_lambda=1.0,
                                               min_child_weight=3, use_label_encoder=False)
                    self.dataset_path = ''
                else:
                    self.dataset_path = dataset_path
                    print 'Initializing model from: ', self.dataset_path
                    self.build_model()
            except NameError:
                print 'XGBoost is not installed. Install with: pip install xgboost'
                exit(1)
        elif self.classifier == 'lgbm':
            try:
                if dataset_path is None:
                    self.clf = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.01, 
                                               max_depth=3, num_leaves=31, subsample=0.8,
                                               colsample_bytree=0.8, reg_alpha=0.1, 
                                               reg_lambda=1.0, min_child_samples=5,
                                               verbose=-1, is_unbalance=True)
                    self.dataset_path = ''
                else:
                    self.dataset_path = dataset_path
                    print 'Initializing model from: ', self.dataset_path
                    self.build_model()
            except NameError:
                print 'LightGBM is not installed. Install with: pip install lightgbm'
                exit(1)
        elif self.classifier == 'lgbm_clf':
            try:
                if dataset_path is None:
                    self.clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01, 
                                                max_depth=3, num_leaves=31, subsample=0.8,
                                                colsample_bytree=0.8, reg_alpha=0.1, 
                                                reg_lambda=1.0, min_child_samples=5,
                                                verbose=-1, class_weight='balanced',
                                                objective='binary', metric='binary_logloss')
                    self.dataset_path = ''
                else:
                    self.dataset_path = dataset_path
                    print 'Initializing model from: ', self.dataset_path
                    self.build_model()
            except NameError:
                print 'LightGBM is not installed. Install with: pip install lightgbm'
                exit(1)
        else:
            print 'Classifier is not supported'

        if model_dir is not None:
            self.model_file_path = model_dir + '_' + model_file_name
        else:
            model_dir = tempfile.mkdtemp()
            self.model_file_path = os.path.join(model_dir, model_file_name)
        print 'Model is saved here: {}'.format(self.model_file_path)

        # if self.classifier == 'sgd':
        #     self.clf = SGDRegressor(max_iter=1000, alpha=1, penalty='l1')

    def load_model(self):
        try:
            self.clf = pickle.load(open(self.model_file_path, 'rb'))
        except:
            print "Could not load model from {}".format(self.model_file_path)
            return False

    def build_model(self):
        if not os.path.exists(self.dataset_path):
            print 'dataset does not exist'
            exit(1)

        self.find_optimal_param()
        self.model_construction()

    def model_construction(self):
        if self.classifier == 'rf':
            self.clf = RandomForestRegressor(n_estimators=self.best_model_params['n_estimators'],
                                            max_depth=self.best_model_params['max_depth'])
        elif self.classifier == 'rf_clf':
            self.clf = RandomForestClassifier(
                n_estimators=self.best_model_params.get('n_estimators', 50),
                max_depth=self.best_model_params.get('max_depth', 4),
                class_weight='balanced')
        elif self.classifier == 'svr' or self.classifier == 'svr_rbf':
            self.clf = SVR(kernel=self.best_model_params.get('kernel', 'rbf'),
                          C=self.best_model_params.get('C', 1.0),
                          gamma=self.best_model_params.get('gamma', 'scale'))
        elif self.classifier == 'bayesian':
            self.clf = BayesianRidge(n_iter=self.best_model_params.get('n_iter', 300),
                                    alpha_1=self.best_model_params.get('alpha_1', 1e-6),
                                    alpha_2=self.best_model_params.get('alpha_2', 1e-6),
                                    lambda_1=self.best_model_params.get('lambda_1', 1e-6),
                                    lambda_2=self.best_model_params.get('lambda_2', 1e-6))
        elif self.classifier == 'gbr':
            self.clf = GradientBoostingRegressor(
                n_estimators=self.best_model_params.get('n_estimators', 100),
                learning_rate=self.best_model_params.get('learning_rate', 0.1),
                max_depth=self.best_model_params.get('max_depth', 3),
                min_samples_split=self.best_model_params.get('min_samples_split', 2),
                min_samples_leaf=self.best_model_params.get('min_samples_leaf', 1),
                subsample=self.best_model_params.get('subsample', 0.8))
        elif self.classifier == 'xgb':
            self.clf = xgb.XGBRegressor(
                n_estimators=self.best_model_params.get('n_estimators', 100),
                learning_rate=self.best_model_params.get('learning_rate', 0.01),
                max_depth=self.best_model_params.get('max_depth', 3),
                subsample=self.best_model_params.get('subsample', 0.8),
                colsample_bytree=self.best_model_params.get('colsample_bytree', 0.8),
                min_child_weight=self.best_model_params.get('min_child_weight', 3),
                reg_alpha=self.best_model_params.get('reg_alpha', 0.1),
                reg_lambda=self.best_model_params.get('reg_lambda', 1.0),
                scale_pos_weight=self.best_model_params.get('scale_pos_weight', 20.0))
        elif self.classifier == 'xgb_clf':
            self.clf = xgb.XGBClassifier(
                n_estimators=self.best_model_params.get('n_estimators', 100),
                learning_rate=self.best_model_params.get('learning_rate', 0.01),
                max_depth=self.best_model_params.get('max_depth', 3),
                subsample=self.best_model_params.get('subsample', 0.8),
                colsample_bytree=self.best_model_params.get('colsample_bytree', 0.8),
                min_child_weight=self.best_model_params.get('min_child_weight', 3),
                reg_alpha=self.best_model_params.get('reg_alpha', 0.1),
                reg_lambda=self.best_model_params.get('reg_lambda', 1.0),
                scale_pos_weight=self.best_model_params.get('scale_pos_weight', 20.0),
                use_label_encoder=False)
        elif self.classifier == 'lgbm':
            self.clf = lgb.LGBMRegressor(
                n_estimators=self.best_model_params.get('n_estimators', 100),
                learning_rate=self.best_model_params.get('learning_rate', 0.01),
                max_depth=self.best_model_params.get('max_depth', 3),
                num_leaves=self.best_model_params.get('num_leaves', 31),
                subsample=self.best_model_params.get('subsample', 0.8),
                colsample_bytree=self.best_model_params.get('colsample_bytree', 0.8),
                reg_alpha=self.best_model_params.get('reg_alpha', 0.1),
                reg_lambda=self.best_model_params.get('reg_lambda', 1.0),
                min_child_samples=self.best_model_params.get('min_child_samples', 5),
                is_unbalance=self.best_model_params.get('is_unbalance', True))
        elif self.classifier == 'lgbm_clf':
            self.clf = lgb.LGBMClassifier(
                n_estimators=self.best_model_params.get('n_estimators', 100),
                learning_rate=self.best_model_params.get('learning_rate', 0.01),
                max_depth=self.best_model_params.get('max_depth', 3),
                num_leaves=self.best_model_params.get('num_leaves', 31),
                subsample=self.best_model_params.get('subsample', 0.8),
                colsample_bytree=self.best_model_params.get('colsample_bytree', 0.8),
                reg_alpha=self.best_model_params.get('reg_alpha', 0.1),
                reg_lambda=self.best_model_params.get('reg_lambda', 1.0),
                min_child_samples=self.best_model_params.get('min_child_samples', 5),
                class_weight='balanced',
                objective='binary', metric='binary_logloss')
                
        self.update_model(features=[], labels=[])

    def predict(self, features):
        try:
            if not hasattr(self, '_feature_columns'):
                self._feature_columns = self.columns
            
            if self.is_classifier:
                if self.classifier == 'xgb_clf':
                    if hasattr(self.clf, 'feature_names_in_'):
                        features_array = np.asarray(features[0]) if isinstance(features[0], (list, np.ndarray)) else \
                                       np.array([features])
                        probs = self.clf.predict_proba(features_array.reshape(1, -1))
                        prob = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                        return prob
                    else:
                        probs = self.clf.predict_proba(np.array(features).reshape(1, -1))
                        prob = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                        return prob
                else:
                    features_array = np.asarray(features[0]) if isinstance(features[0], (list, np.ndarray)) else \
                                   np.array([features])
                    probs = self.clf.predict_proba(features_array.reshape(1, -1))
                    prob = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                    return prob
            
            if self.classifier == 'xgb':
                if hasattr(self.clf, 'feature_names_in_'):
                    features_array = np.asarray(features[0]) if isinstance(features[0], (list, np.ndarray)) else \
                                   np.array([features])
                    return self.clf.predict(features_array.reshape(1, -1))[0]
            return self.clf.predict(np.array(features).reshape(1, -1))[0]
        except sklearn.exceptions.NotFittedError:
            if self.is_classifier:
                return self._heuristic_predict(features)
            else:
                print 'The model is not fitted yet.'
                return sum(features[0])
        except xgb.core.XGBoostError as e:
            print 'XGBoost error: {}'.format(e)
            if self.is_classifier:
                return self._heuristic_predict(features)
            else:
                return sum(features[0])
        except Exception as e:
            print 'Prediction error: {}'.format(e)
            try:
                features_array = np.asarray(features[0]) if isinstance(features[0], (list, np.ndarray)) else \
                               np.array([features])
                if self.is_classifier:
                    probs = self.clf.predict_proba(features_array.reshape(1, -1))
                    return probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                else:
                    return self.clf.predict(features_array.reshape(1, -1))[0]
            except Exception as e2:
                print 'Feature conversion failed: {}'.format(e2)
                if self.is_classifier:
                    return self._heuristic_predict(features)
                else:
                    return sum(features[0])

    def _heuristic_predict(self, features):
        """Heuristic prediction method when classifier model is not initialized"""
        try:
            features = np.array(features)
            if isinstance(features[0], (list, np.ndarray)):
                features_array = np.asarray(features[0])
            else:
                features_array = np.array(features)
            
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            weights = np.array([2.0, 1.5, 3.0, 4.0, 0.5] + [1.0] * (len(features_array[0]) - 5))
            weights = weights[:len(features_array[0])]
            
            weighted_sum = np.dot(features_array[0], weights)
            
            normalized_score = weighted_sum / len(features_array[0])
            prob = 1.0 / (1.0 + np.exp(-normalized_score / 2.0))
            
            return float(max(0.0, min(1.0, prob)))
        except Exception as e2:
            return 0.5

    def update_model(self, features, labels):
        features_dict_list = list()
        if len(self.labels) == 0:
            self.labels = np.array(labels).reshape(-1, 1)
        else:
            if len(labels) > 0:
                self.labels = np.concatenate((self.labels, np.array(labels).reshape(-1, 1)))
            else:
                self.labels = np.array(self.labels).reshape(-1, 1)
        
        if self.is_classifier and len(labels) > 0:
            self.labels = (self.labels > 0).astype(int)

        for feature in features:
            features_dict_list.append(dict(zip(self.columns, feature)))
        if self.features.shape[0] == 0:
            self.features = pd.DataFrame(features_dict_list)
        else:
            self.features = pd.concat([self.features, pd.DataFrame(features_dict_list)])
        
        # Calculate class balance for use in scale_pos_weight
        if len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            if len(unique_labels) > 1:
                neg_count = counts[0]
                pos_count = counts[1]
                if pos_count > 0:  # Prevent division by zero
                    self.scale_pos_weight = float(neg_count) / pos_count
        
        try:
            self.clf.fit(X=self.features, y=self.labels.ravel())
        except Exception as e:
            print "Error fitting model: {}".format(e)

    def remove_model(self, model_file_path=''):
        if model_file_path == '':
            os.remove(self.model_file_path)
        else:
            os.remove(model_file_path)

    def save_model(self):
        pickle.dump(self.clf, open(self.model_file_path, 'wb'))
    
    

    def find_optimal_param(self):
        dataset = pd.read_csv(self.dataset_path)
        self.labels = dataset.label
        
        if self.is_classifier:
            self.labels = (self.labels > 0).astype(int)
            positive_ratio = np.sum(self.labels) / float(len(self.labels))
            print "Converting labels to binary classification - positive samples: {:.2f}%".format(positive_ratio * 100)
        
        # Calculate class balance for models that need it
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        if len(unique_labels) > 1:
            neg_count = counts[0]
            pos_count = counts[1]
            if pos_count > 0:  # Prevent division by zero
                self.scale_pos_weight = float(neg_count) / pos_count
                if self.is_classifier:
                    print "Class balance - negative:positive = {:.2f}:1".format(self.scale_pos_weight)

        try:
            dataset.drop('window', axis=1, inplace=True)
        except:
            pass  # Column may not exist
        dataset.drop('label', axis=1, inplace=True)
        try:
            dataset.drop('id', axis=1, inplace=True)
        except:
            pass  # Column may not exist
            
        self.features = dataset
        
        # Select the appropriate parameter grid based on classifier
        if self.classifier == 'rf':
            grid = self.get_rfregressor_params()
        elif self.classifier == 'rf_clf':
            grid = self.get_rfclassifier_params()
        elif self.classifier == 'svr':
            grid = self.get_svregressor_params()
        elif self.classifier == 'svr_rbf':
            grid = self.get_svr_rbf_params()
        elif self.classifier == 'bayesian':
            grid = self.get_bayesian_params()
        elif self.classifier == 'gbr':
            grid = self.get_gbregressor_params()
        elif self.classifier == 'xgb':
            grid = self.get_xgbregressor_params()
        elif self.classifier == 'xgb_clf':
            grid = self.get_xgbclassifier_params()
        elif self.classifier == 'lgbm':
            grid = self.get_lgbmregressor_params()
        elif self.classifier == 'lgbm_clf':
            grid = self.get_lgbmclassifier_params()
        else:
            print "Unsupported classifier for parameter optimization"
            return

        cv = min(5, max(2, int(len(self.labels) / 10)))
        
        scoring = 'roc_auc' if self.is_classifier else 'neg_mean_squared_error'
        
        # For very small datasets, use RandomizedSearchCV instead of exhaustive GridSearchCV
        if len(self.labels) < 50 and self.classifier not in ['bayesian', 'svr', 'svr_rbf']:
            gd_sr = RandomizedSearchCV(estimator=grid['clf'],
                                    param_distributions=grid['grid_param'],
                                    scoring=scoring,
                                    cv=cv,
                                    n_jobs=-1,
                                    n_iter=10)  # Limit number of iterations for faster results
        else:
            gd_sr = GridSearchCV(estimator=grid['clf'],
                                param_grid=grid['grid_param'],
                                scoring=scoring,
                                cv=cv,
                                n_jobs=-1)
                                
        gd_sr.fit(self.features, self.labels)
        print grid['name'], gd_sr.best_params_, 'Score: ', gd_sr.best_score_
        self.best_model_params = gd_sr.best_params_

    def get_rfregressor_params(self):
        grid_param = {
            'n_estimators': [10, 20, 50, 70],
            'max_depth': [3, 4, 5, 6]
        }
        clf = RandomForestRegressor()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'rfreg'}

    def get_rfclassifier_params(self):
        grid_param = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        clf = RandomForestClassifier(class_weight='balanced')
        return {'clf': clf, 'grid_param': grid_param, 'name': 'rf_clf'}

    def get_svregressor_params(self):
        grid_param = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
        clf = SVR()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'svreg'}
        
    def get_svr_rbf_params(self):
        grid_param = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        clf = SVR(kernel='rbf')
        return {'clf': clf, 'grid_param': grid_param, 'name': 'svr_rbf'}
        
    def get_bayesian_params(self):
        grid_param = {
            'n_iter': [100, 300, 500],
            'alpha_1': [1e-7, 1e-6, 1e-5],
            'alpha_2': [1e-7, 1e-6, 1e-5],
            'lambda_1': [1e-7, 1e-6, 1e-5],
            'lambda_2': [1e-7, 1e-6, 1e-5]
        }
        clf = BayesianRidge()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'bayesian'}
        
    def get_gbregressor_params(self):
        grid_param = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }
        clf = GradientBoostingRegressor()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'gbr'}
        
    def get_xgbregressor_params(self):
        grid_param = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.005, 0.01, 0.05],
            'max_depth': [2, 3, 4],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0.01, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0],
            'scale_pos_weight': [10, 20, 50]
        }
        clf = xgb.XGBRegressor()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'xgb'}
        
    def get_xgbclassifier_params(self):
        grid_param = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.005, 0.01, 0.05],
            'max_depth': [2, 3, 4],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'scale_pos_weight': [1, 5, 10, 20]
        }
        clf = xgb.XGBClassifier(use_label_encoder=False)
        return {'clf': clf, 'grid_param': grid_param, 'name': 'xgb_clf'}
        
    def get_lgbmregressor_params(self):
        grid_param = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.005, 0.01, 0.05],
            'max_depth': [2, 3, 4],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [3, 5, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0.01, 0.1, 1.0],
            'reg_lambda': [0.1, 1.0],
            'is_unbalance': [True]
        }
        clf = lgb.LGBMRegressor()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'lgbm'}

    def get_lgbmclassifier_params(self):
        grid_param = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.005, 0.01, 0.05],
            'max_depth': [2, 3, 4],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [3, 5, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'class_weight': ['balanced']
        }
        clf = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
        return {'clf': clf, 'grid_param': grid_param, 'name': 'lgbm_clf'}

    def get_corr(self, output='/tmp/corr_mat.pdf'):
        dataset = pd.read_csv(self.dataset_path)
        plt.figure(figsize=(25, 20))
        sb.heatmap(dataset.corr(), annot=True, cmap=sb.diverging_palette(20, 220, n=200))
        plt.savefig(output, pad_inches=0)

    @staticmethod
    def get_feature_importance(dataset_dir='', boxplot_path=''):
        list_of_files = list()
        for (dir_path, dir_names, file_names) in os.walk(dataset_dir):
            list_of_files += [os.path.join(dir_path, file_name) for file_name in file_names
                              if file_name.endswith('rf_data.csv')]

        clf = RandomForestRegressor(n_estimators=100, max_depth=4)
        features_importance_all = []
        for file_path in list_of_files:
            dataset = pd.read_csv(file_path)
            y = dataset.label
            dataset.drop('label', axis=1, inplace=True)
            dataset.drop('id', axis=1, inplace=True)
            x = dataset
            clf.fit(X=x, y=y)
            features_name = dataset.columns.values
            features_importance = {}
            for index, feature_importance in enumerate(clf.feature_importances_):
                features_importance[features_name[index].title()] = feature_importance

            features_importance_all.append(features_importance)
        feature_importance_data = pd.DataFrame(features_importance_all)
        median = feature_importance_data.median()
        median.sort_values(ascending=False, inplace=True)
        feature_importance_data = feature_importance_data[median.index]
        # plt.figure(figsize=(20, 10))
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.3)
        plt.grid()
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Gini Importance', fontsize=14)
        feature_importance_data.boxplot(rot=45)
        plt.savefig(boxplot_path, bbox_inches='tight', pad_inches=0)


def testRandomForest():
    print "TEST RF"
    mlengine = MLEngine()
    predicted_value = mlengine.predict([[0,2,1,5,4,3,5,5,3,2,2,2,2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0,2,1,6,3,4,2,4,6,4,4,4,5]], [[200]])
    predicted_value = mlengine.predict([[0,2,1,8,7,5,5,3,6,7,5,5,5]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2,8,6,5,4,6,5,2,2,2,5]], [[143]])
    predicted_value = mlengine.predict([[2,3,2,7,4,5,6,9,0,6,5,5,5]])
    print "predicted value2: ", predicted_value


def testRandomForestInit():
    print "TEST RF INIT"
    mlengine = MLEngine(dataset_path='/home/eric/work/savior/newtcpdump_data.csv')
    predicted_value = mlengine.predict([[0,2,1,3,2,4,2,4,5,1,5,5,5]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[2,2,1,6,8,7,8,4,5,3,5,5,5,5]], [[200]])
    predicted_value = mlengine.predict([[0,2,1,7,8,4,4,5,2,3,1,6,6,6]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[6,6,1,3,2,6,4,8,5,3,1,5,5]], [[143]])
    predicted_value = mlengine.predict([[5,6,2,6,3,2,5,6,3,4,1,6,6,6]])
    print "predicted value2: ", predicted_value


def testSVM():
    print "TEST SVM"
    mlengine = MLEngine(classifier='svr')
    predicted_value = mlengine.predict([[0, 2, 1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1, 3, 2]], [[143]])
    predicted_value = mlengine.predict([[2, 3, 2]])
    print "predicted value2: ", predicted_value


def testSVMInit():
    print "TEST SVM INIT"
    mlengine = MLEngine(classifier='svr',
                        dataset_path='/home/eric/work/savior/newtcpdump_data.csv')
    predicted_value = mlengine.predict([[0, 2, 1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1, 3, 2]], [[143]])
    predicted_value = mlengine.predict([[2, 3, 2]])
    print "predicted value2: ", predicted_value


def testBayesian():
    print "TEST Bayesian Ridge"
    mlengine = MLEngine(classifier='bayesian')
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value1 : ", predicted_value


def testGBR():
    print "TEST Gradient Boosting Regressor"
    mlengine = MLEngine(classifier='gbr')
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value1 : ", predicted_value


def testXGB():
    print "TEST XGBoost Regressor"
    mlengine = MLEngine(classifier='xgb')
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value1 : ", predicted_value


def testLGBM():
    print "TEST LightGBM Regressor"
    mlengine = MLEngine(classifier='lgbm')
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]], [[200]])
    predicted_value = mlengine.predict([[0, 2, 1, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2]])
    print "predicted value1 : ", predicted_value


def test_corr():
    mlengine = MLEngine(dataset_path='/tmp/tcpdump_data.csv')
    mlengine.get_corr("/tmp/corr_tcpdump.pdf")


def test_feature_importance():
    MLEngine.get_feature_importance('/Users/mansourahmadi/Bank/Work/NEU/MEUZZ/meuzz-learning-data',
                                    '/tmp/feature_importance.pdf')


if __name__ == "__main__":
    testRandomForest()
    # testRandomForestInit()
    # testSVM()
    # testSVMInit()
    # testBayesian()
    # testGBR()
    # testXGB()
    # testLGBM()
    # test_corr()
    # test_feature_importance()
