"""
PFE Boulet Olgiati
"""

import json
import time
import numpy as np
from sklearn.metrics import accuracy_score

import data
from utils import *
from model import ESN



if __name__=='__main__' : 

    with open("config.json", "r") as f:
        config = json.load(f)

    SEED = config["SEED"]
    config_data = config["data"]
    config_training = config["training"]

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

    ################################################################
    # TRAIN MODEL
    ################################################################

    start = time.time()
    print("\nTraining a single ESN ...")
    esn = ESN(n_inputs=img_size, n_outputs=n_outputs, n_reservoir=n_reservoir,
        alpha=alpha, spectral_radius=spectral_radius, sparsity=sparsity, noise=noise, 
        input_scaling=input_scaling, input_shift=input_shift, 
        teacher_scaling=teacher_scaling, teacher_shift=teacher_shift, 
        teacher_forcing=teacher_forcing, feedback_scaling=feedback_scaling, 
        out_activation=out_activation, inverse_out_activation=inverse_out_activation,
        use_gradient_descent=use_gradient_descent, learning_rate=learning_rate, 
        l2_rate=l2_rate, n_epochs=n_epochs, batch_size=batch_size, 
        use_rls=use_rls, forget_rate=forget_rate)
    train_preds = esn.fit(U_train[:n_training], V_train[:n_training])
    end = time.time()
    print("\nTraining executed in %s seconds." % (end-start))

    print("\nEvaluating on the test set ...")
    preds = esn.predict(U_test[:n_test], continuation=True)
    test_acc = accuracy_score(V_test[:int(n_test/img_size)], preds[:int(n_test/img_size)])
    print("Accuracy on the test set : ", test_acc)