import xgboost as xgb 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score  
import time

dp_per_tree = 0.25
n_trees = 20
subsample = 0.2

# we transform the classification problem into a regression one 
obj = 'reg:squarederror'

x, y = make_classification(n_samples = 100000, n_features = 50, random_state = 100) 

base_score = 0.5
scorefunc = mean_squared_error 

total_budget_spent = n_trees * np.log(1 + subsample*(np.exp(dp_per_tree) - 1))
print('Total epsilon spent ', total_budget_spent) 

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2) 

n_data = trainX.shape[0]
n_features = trainX.shape[1]

feature_min = []
feature_max = []

# we need the feature bounds to build the DMatrix for DP training 
for i in range(n_features):
    feature_min.append( min(trainX[:,i]) )
for i in range(n_features): 
    feature_max.append( max(trainX[:,i]) )

dtrain = xgb.DMatrix(trainX, label=trainY, feature_min=feature_min,
    feature_max=feature_max) 

dtest = xgb.DMatrix(testX, label=testY, feature_min=feature_min,
    feature_max=feature_max) 

print('DMatrix built')

paramsDP =  {'objective': obj,
        'tree_method':'approxDP', # this is Sarus XGBoost tree updater 
        'dp_epsilon_per_tree': dp_per_tree,
        'max_depth': 6,
        #'verbosity' : 3,
        'learning_rate' : 0.2,
        'lambda' : 0.1,
        'base_score' : base_score,
        'subsample' : subsample,
        'min_child_weight' : 500,
        'nthread' : 4}

paramsNonDP =  {'objective': obj,
        'tree_method':'approx',
        'max_depth': 6,
        'learning_rate' : 0.7,
        'lambda' : 0.1,
        'base_score' : base_score,
        'subsample' : subsample, 
        'min_child_weight' : 100,
        'nthread' : 4} 

begin = time.time() 
bstDP = xgb.train(paramsDP, dtrain, num_boost_round=n_trees) 
end = time.time() 

runtime_dp = end - begin

begin = time.time() 
bst = xgb.train(paramsNonDP, dtrain, num_boost_round=n_trees)
end = time.time() 

runtime_non_dp = end - begin 

predDP = bstDP.predict(dtest)
predNonDP = bst.predict(dtest)

test_errors_dp =  100 * (1-accuracy_score(testY, predDP > 0.5))
test_errors_non_dp =  100 * (1-accuracy_score(testY, predNonDP > 0.5))

print('test error DP', test_errors_dp)
print('test error non-DP', test_errors_non_dp)
print('runtime DP', runtime_dp)
print('runtime non DP', runtime_non_dp)

# bstDP.save_model('testDP.json')
# bst.save_model('test.json')