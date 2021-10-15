import xgboost as xgb 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file,dump_svmlight_file, make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score  
import time 

def run_experiment(name, dp_per_tree = 0.5, n_trees = 20, subsample = 0.2, n_runs = 20, lr=0.8): 
    obj = 'reg:squarederror'
    multiplier = 1 

    scorefunc = accuracy_score
 
    if name == 'covtype':
        df = load_svmlight_file('covtype.libsvm.binary.scale')
        x,y = df
        x = x[0:100000,:].toarray()
        y = y[0:100000]
        y = y-1 # classes are 0, 1 
        base_score = 0.5

    elif name == 'cod-rna':
        base_score = 0.5 
        df = load_svmlight_file('cod-rna.libsvm')
        x,y = df 
        y = 0.5*(y+1) 
        x = x.toarray() 

    elif name == "YearPredictionMSD":
        # multiclass 
        df = pd.read_csv('YearPredictionMSD.txt', nrows=100000) 
        x = df.values[:, 1:] 
        y = df.values[:, 0] 

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(y.reshape(-1,1))
        y = scaler.transform(y.reshape(-1,1))
        multiplier = (scaler.data_max_ - scaler.data_min_) / 2
        base_score = 0.0 
        lr = 0.1

    elif name == 'adult': 
        # target is binary -1 or 1 
        df = load_svmlight_file('adult.libsvm')
        x,y = df 
        y = 0.5*(y+1)
        x = x.toarray()
        base_score = 0.5
        lr = 0.5
    elif name == 'abalone': 
        df = load_svmlight_file('abalone') 
        x,y = df
        x = x.toarray() 

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(y.reshape(-1,1))
        y = scaler.transform(y.reshape(-1,1))
        multiplier = (scaler.data_max_ - scaler.data_min_) / 2
        base_score = 0.0 

    elif name == 'synthetic_reg': 
        x, y = make_regression(n_samples = 100000, n_features = 70, random_state = 100) 

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(y.reshape(-1,1))
        y = scaler.transform(y.reshape(-1,1))
        multiplier = (scaler.data_max_ - scaler.data_min_) / 2

        scorefunc = mean_squared_error 
        base_score = 0 

    elif name == 'synthetic_cls':
        x,y = make_classification(n_samples = 100000, n_features = 80, random_state = 100)
        base_score = 0.5 
    
    total_budget_spent = dp_per_tree * n_trees * subsample 
    print('Running experiment with dataset ', name)
    print('total epsilon spent ', total_budget_spent) 

    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2) 

    n_data = trainX.shape[0]
    n_features = trainX.shape[1]

    feature_min = []
    feature_max = []
    # bounds will contain 2*n_features values 
    # corresponding to the range min and max respectively
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
            'learning_rate' : lr,
            'lambda' : 0.1,
            'base_score' : base_score,
            'subsample' : subsample,
            'min_child_weight' : 500,
            'nthread' : 4}

    paramsNonDP =  {'objective': obj,
            'tree_method':'approx',
            'max_depth': 6,
            'learning_rate' : lr,
            'lambda' : 0.1,
            'base_score' : base_score,
            'subsample' : subsample, 
            'min_child_weight' : 1,
            'nthread' : 4}

    test_errors_non_dp = np.zeros(n_runs)
    test_errors_dp = np.zeros(n_runs)
    runtimes_dp = np.zeros(n_runs)
    runtimes_non_dp = np.zeros(n_runs) 

    for k in range(n_runs): 
        print('Run ', k+1, '/', n_runs)

        begin = time.time() 
        bstDP = xgb.train(paramsDP, dtrain, num_boost_round=n_trees) 
        end = time.time() 

        runtimes_dp[k] = end - begin

        begin = time.time() 
        bst = xgb.train(paramsNonDP, dtrain, num_boost_round=n_trees)
        end = time.time() 

        runtimes_non_dp[k] = end - begin 

        predDP = bstDP.predict(dtest)
        predNonDP = bst.predict(dtest)

        if name in ['covtype', 'adult', 'synthetic_cls', 'cod-rna']:
            test_errors_dp[k] = multiplier * accuracy_score(testY, predDP > 0.5)
            test_errors_non_dp[k] = multiplier * accuracy_score(testY, predNonDP > 0.5)

        elif name in ['YearPredictionMSD', 'synthetic_reg', 'abalone']:
            test_errors_dp[k] = multiplier * mean_squared_error(testY, predDP, squared=False)
            test_errors_non_dp[k] = multiplier * mean_squared_error(testY, predNonDP, squared=False)
    return test_errors_dp, test_errors_non_dp, runtimes_dp, runtimes_non_dp

#np.random.seed(123)

def run_all_exp(datasets = ['covtype', 'adult', 'synthetic_cls', 'synthetic_reg', 'abalone'], 
    output='sarus_dp_results.txt'):
    f = open(output, 'a') 

    # epslist = [1, 2, 4, 6, 8, 10] 
    epslist = [0.1, 0.5, 1, 2, 5]
    epslist = np.array(epslist) / 4.0

    for dset in datasets:
        for epsilon_per_tree in epslist: 
            errors_dp, errors, runtimes_dp, runtimes = run_experiment(dset, epsilon_per_tree, 20, 0.2, 5,
                lr=0.6)

            print('Mean DP error: ', np.mean(errors_dp))
            print('Mean Non-DP error: ', np.mean(errors))

            print('Mean DP runtime:', np.mean(runtimes_dp))
            print('Mean Non-DP runtime:', np.mean(runtimes))

            f.write(dset + ' epsilon = ' + str(epsilon_per_tree*4) + 
                ' mean dp error = ' + str(np.mean(errors_dp)) + ' non-DP error = ' + 
                str(np.mean(errors)) + 
                ' mean dp runtime = ' + str(np.mean(runtimes_dp)) + ' non dp runtime = '
                + str(np.mean(runtimes)) + '\n')
            f.flush() 

np.random.seed(121) 

run_all_exp(['covtype'], 'cvov.txt')

#print(run_experiment('adult', 10.0/4, 20, 0.2, 5, lr=0.9))