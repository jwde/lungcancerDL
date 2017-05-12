# lungcancerDL
COMP150DL final project - LUNA 2016 + Kaggle Data Science Bowl 2017 submission
This repository contains the code behind our project

Jay DeStories, Jason Fan, Alex Tong

Running the kernels:

    Lungs: everything for lungs lives in the lungdl folder

        With Docker installed, launch our container by running start.sh

        Define networks in config.py

        Run a network using
            python main.py [parameters]

            Parameters are as follows:
                --DATA_DIR         Path to the data directory
                --LABELS_FILE      Path to the labels csv
                --NET              Name of the network (from config.py) to train
                --MODELS_DIR       Where to store the model and loss history
                --SAVE_NAME        What to call the saved model
                --LOAD_MODEL       (Optional) Name of model to load to 
                                       initialize weights
                --NUM_EPOCHS       Number of epochs to train the model
                --TRAINING_SIZE    How much of the dataset to use for training 
                                       (the rest is used for validation)
                --NO_VAL           (Optional) Don't run the validation step


        Evaluation -- produce AP, loss, and precision/recall curves

            Evaluate baseline models:
                python evalute.py

            Evaluate trained models:
                specify models to evaluate in evaluate_all_models.py
                    An entry in models contains:
                        (model name from config, saved model name, data dir)
                python evaluate_all_models.py


    Ants vs Bees: everything lives in the ants_and_bees folder

        With Docker installed, launch our container by running start.sh

        Define networks in config.py

        Train a network using
            python main.py -t [name of network from config.py]
