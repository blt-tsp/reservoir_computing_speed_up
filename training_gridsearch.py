"""
PFE Boulet Olgiati
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import data
from utils import *
from model import ESN, ESNEstimator



if __name__=='__main__' : 

    with open("config.json", "r") as f:
        config = json.load(f)

    SEED = config["SEED"]
    config_data = config["data"]
    config_training = config["training"]
    config_gridsearch = config["training_gridsearch"]

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

    n_reservoir_range = config_gridsearch["N_RESERVOIR"]
    alpha_range = config_gridsearch["ALPHA"]
    spectral_radius_range = config_gridsearch["SPECTRAL_RADIUS"]
    input_scaling_range = config_gridsearch["INPUT_SCALING"]
    input_shift_range = config_gridsearch["INPUT_SHIFT"]
    teacher_scaling_range = config_gridsearch["TEACHER_SCALING"]
    teacher_shift_range = config_gridsearch["TEACHER_SHIFT"]
    teacher_forcing_range = config_gridsearch["TEACHER_FORCING"]
    feedback_scaling_range = config_gridsearch["FEEDBACK_SCALING"]
    learning_rate_range = config_gridsearch["LEARNING_RATE"]
    l2_rate_range = config_gridsearch["L2_RATE"]
    forget_rate_range = config_gridsearch["FORGET_RATE"]

    ################################################################
    # DEFINE THE GRIDSEARCH OBJECT
    ################################################################

    start = time.time()
    print("\nTraining a GridSearch ...")
    
    esn = ESN(img_size, n_outputs, 
               n_reservoir, alpha, spectral_radius, sparsity, noise,
               input_scaling, input_shift, 
               teacher_scaling, teacher_shift,
               teacher_forcing, feedback_scaling, 
               out_activation, inverse_out_activation,
               use_gradient_descent, learning_rate, l2_rate, n_epochs, batch_size, 
               use_rls, forget_rate)
    scoring_func = make_scorer(esn.score)

    param_grid = {
        'n_reservoir': n_reservoir_range,
        'alpha': alpha_range,
        'spectral_radius': spectral_radius_range, 
        'sparsity': [sparsity], 
        'noise': [noise], 
        'input_scaling': input_scaling_range,
        'input_shift': input_shift_range, 
        'teacher_forcing': teacher_forcing_range, 
        'teacher_scaling': teacher_scaling_range, 
        'teacher_shift': teacher_shift_range, 
        'feedback_scaling': feedback_scaling_range,
        'out_activation': [out_activation], 
        'inverse_out_activation': [inverse_out_activation], 
        'random_state': [None], 
        'use_gradient_descent': [use_gradient_descent],
        'learning_rate': learning_rate_range, 
        'l2_rate': l2_rate_range, 
        'use_rls': [use_rls], 
        'forget_rate': forget_rate_range,
        'n_epochs': [n_epochs], 
        'batch_size': [batch_size]
    }

    gs = GridSearchCV(ESNEstimator(img_size, n_outputs), param_grid=param_grid, scoring=scoring_func)

    ################################################################
    # TRAIN THE GRIDSEARCH
    ################################################################

    gs.fit(U_train[:n_training], V_train[:n_training])
    results = pd.DataFrame(gs.cv_results_).drop(columns=["param_inverse_out_activation", \
                                                "param_out_activation", "params", "split0_test_score", \
                                                "split1_test_score", "split2_test_score", "split3_test_score", \
                                                "split4_test_score", "mean_fit_time", "std_fit_time", \
                                                "mean_score_time", "std_score_time"])
    results = results.sort_values(by="rank_test_score")
    print("\n5 first lines of the results dataframe : ")
    print(results.head(5))
    print("\nBest score : ", gs.best_score_)
    print("Best params : \n", gs.best_params_)

    ################################################################
    # TRAIN AND EVALUATE THE BEST MODEL
    ################################################################

    print("\nTraining the best model ...")
    best_esn = ESN(n_inputs = img_size, n_outputs = n_outputs, 
                n_reservoir = gs.best_params_['n_reservoir'], 
                alpha = gs.best_params_['alpha'], 
                spectral_radius = gs.best_params_['spectral_radius'], 
                sparsity = gs.best_params_['sparsity'], 
                noise = gs.best_params_['noise'], 
                input_scaling = gs.best_params_['input_scaling'], 
                input_shift = gs.best_params_['input_shift'], 
                teacher_forcing = gs.best_params_['teacher_forcing'], 
                teacher_scaling = gs.best_params_['teacher_scaling'], 
                teacher_shift = gs.best_params_['teacher_shift'], 
                feedback_scaling = gs.best_params_['feedback_scaling'], 
                out_activation = gs.best_params_['out_activation'], 
                inverse_out_activation = gs.best_params_['inverse_out_activation'], 
                random_state = gs.best_params_['random_state'], 
                use_gradient_descent = gs.best_params_['use_gradient_descent'], 
                learning_rate = gs.best_params_['learning_rate'], 
                l2_rate = gs.best_params_['l2_rate'], 
                use_rls = gs.best_params_['use_rls'], 
                forget_rate = gs.best_params_['forget_rate'],
                n_epochs = gs.best_params_['n_epochs'], 
                batch_size = gs.best_params_['batch_size']
    )
    train_preds = best_esn.fit(U_train[:n_training], V_train[:n_training])
    end = time.time()
    print("\nTraining executed in %s seconds." % (end-start))

    print("\nEvaluating on the test set ...")
    preds = best_esn.predict(U_test[:n_test], continuation=True)
    print("Accuracy on the test set : ", accuracy_score(V_test[:int(n_test/img_size)], preds[:int(n_test/img_size)]))