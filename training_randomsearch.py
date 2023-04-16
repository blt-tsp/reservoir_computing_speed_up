"""
PFE Boulet Olgiati
"""

import json
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV  
from scipy.stats import uniform, norm

import data
from utils import *
from model import ESN, ESNEstimator



if __name__=='__main__' : 

    with open("config.json", "r") as f:
        config = json.load(f)

    SEED = config["SEED"]
    config_data = config["data"]
    config_training = config["training"]
    config_randomsearch = config["training_randomsearch"]

    ################################################################
    # RETRIEVE TRAINING AND TEST DATA
    ################################################################

    data_path = config_data["DATA_PATH"]

    U_train, V_train, U_test, V_test = data.main(data_path)

    print("Shape of the training set : \t\t", np.shape(U_train))
    print("Shape of the training labels set : \t", np.shape(V_train))
    print("Shape of the test set : \t\t", np.shape(U_test))
    print("Shape of the test labels set : \t\t", np.shape(V_test))

    ################################################################
    # CONFIG TRAINING PARAMETERS
    ################################################################

    img_size = config_training["IMG_SIZE"]
    n_outputs = config_training["N_OUTPUTS"]
    n_reservoir = config_training["N_RESERVOIR"]
    alpha = config_training["ALPHA"]
    spectral_radius = config_training["SPECTRAL_RADIUS"]
    sparsity = config_training["SPARSITY"]
    noise = config_training["NOISE"]
    input_scaling = config_training["INPUT_SCALING"]
    input_shift = config_training["INPUT_SHIFT"]
    teacher_scaling = config_training["TEACHER_SCALING"]
    teacher_shift = config_training["TEACHER_SHIFT"]
    teacher_forcing = config_training["TEACHER_FORCING"]
    feedback_scaling = config_training["FEEDBACK_SCALING"]
    if config_training["OUT_ACTIVATION"]=='identity' : 
        out_activation = identity
    elif config_training["OUT_ACTIVATION"]=='tanh' : 
        out_activation = tanh
    if config_training["INVERSE_OUT_ACTIVATION"]=='identity' : 
        inverse_out_activation = identity
    elif config_training["INVERSE_OUT_ACTIVATION"]=='tanh' : 
        inverse_out_activation = tanh
    use_gradient_descent = config_training["USE_GRADIENT_DESCENT"]
    learning_rate = config_training["LEARNING_RATE"]
    l2_rate = config_training["L2_RATE"]
    n_epochs = config_training["N_EPOCHS"]
    batch_size = config_training["BATCH_SIZE"]
    use_rls = config_training["USE_RLS"]
    forget_rate = config_training["FORGET_RATE"]
    n_training = config_training["N_TRAINING"]
    n_test = config_training["N_TEST"]

    n_reservoir_range = config_randomsearch["N_RESERVOIR"]
    teacher_forcing_range = config_randomsearch["TEACHER_FORCING"]
    forget_rate_range = config_randomsearch["FORGET_RATE"]

    ################################################################
    # DEFINE THE RANDOMSEARCH OBJECT
    ################################################################

    start = time.time()
    print("\nTraining a RandomSearch ...")
    esn = ESNEstimator(n_inputs=28, n_outputs=10)
    scoring_func = make_scorer(esn.score)
    distrib = dict(n_reservoir = n_reservoir_range, 
                alpha = norm(0.0001, 0.001), 
                spectral_radius = uniform(0, 1), 
                sparsity = [sparsity], 
                noise = [noise],  
                input_scaling = uniform(0,1), 
                input_shift = uniform(0,1), 
                teacher_scaling = uniform(0,1), 
                teacher_shift = uniform(0,1), 
                teacher_forcing = teacher_forcing_range, 
                random_state = [None], 
                feedback_scaling = uniform(0,1),
                out_activation = [identity],  
                inverse_out_activation = [identity], 
                use_gradient_descent = [use_gradient_descent],
                learning_rate = norm(10e-5, 10e-6), 
                l2_rate = norm(10e-5, 10e-6), 
                use_rls = [use_rls], 
                forget_rate = forget_rate_range, 
                n_epochs = [n_epochs], 
                batch_size = [batch_size])
    clf = RandomizedSearchCV(esn, distrib, random_state=0, scoring=scoring_func)
    
    ################################################################
    # TRAIN THE RANDOM SEARCH
    ################################################################

    search = clf.fit(U_train[:n_training], V_train[:n_training])

    ################################################################
    # TRAIN AND EVALUATE THE BEST MODEL
    ################################################################

    print("\nTraining the best model found ...")
    best_esn = ESN(n_inputs = 28, n_outputs = 10, 
            n_reservoir = search.best_params_['n_reservoir'],
            alpha = search.best_params_['alpha'], 
            spectral_radius = search.best_params_['spectral_radius'], 
            sparsity = search.best_params_['sparsity'], 
            noise = search.best_params_['noise'], 
            input_scaling = search.best_params_['input_scaling'], 
            input_shift = search.best_params_['input_shift'], 
            teacher_forcing = search.best_params_['teacher_forcing'], 
            teacher_scaling = search.best_params_['teacher_scaling'], 
            teacher_shift = search.best_params_['teacher_shift'], 
            feedback_scaling = search.best_params_['feedback_scaling'], 
            out_activation = search.best_params_['out_activation'], 
            inverse_out_activation = search.best_params_['inverse_out_activation'], 
            random_state = search.best_params_['random_state'], 
            use_gradient_descent = search.best_params_['use_gradient_descent'], 
            learning_rate = search.best_params_['learning_rate'], 
            l2_rate = search.best_params_['l2_rate'], 
            use_rls = search.best_params_['use_rls'], 
            forget_rate = search.best_params_['forget_rate'],
            n_epochs = search.best_params_['n_epochs'], 
            batch_size = search.best_params_['batch_size']
    )
    train_preds = best_esn.fit(U_train[:n_training], V_train[:n_training])
    end = time.time()
    print("\nTraining executed in %s seconds." % (end-start))

    print("\nEvaluating on the test set ...")
    preds = best_esn.predict(U_test[:n_test], continuation=True)
    print("Accuracy on the test set : ", accuracy_score(V_test[:int(n_test/img_size)], preds[:int(n_test/img_size)]))
        