"""
PFE Boulet Olgiati
"""

1. To train a single ESN : 
    - modify the training parameters in the file config.json under "training"
    - run the file training.py

2. To train a GridSearch : 
    - modify the constant training parameters in the file config.json under "training"
    - modify the variable training parameters' ranges in the file config.json under "training_gridsearch"
    - run the file training_gridsearch.py

3. To train a RandomSearch : 
    - modify the constant training parameters in the file config.json under "training"
    - modify the variable training parameters' ranges in the file config.json under "training_randomsearch"
    - run the file training_randomsearch.py