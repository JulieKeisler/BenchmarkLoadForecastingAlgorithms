# Benchmark-electric-power-consumption-forecasting-algorithms

This repository corresponds to a project made at ETH Zürich in the Power System Laboratory.

**Abstract**

Electricity consumption forecasting is today an important topic for electricity efficiency management. A lot of research has been done on the subject but with different datasets and setup. A plethora of algorithms using deep learning, statistical or signal processing tools have been created without any real benchmark. This semester project aims at creating a benchmark testifying the performance of a panel of models on a specific  load consumption dataset. The second part of the work dig into how taking full advantage of the strengths and weaknesses of these individual models to create combinatory models that outperform the individual ones. More specifically we looked at a methodology based on divide and conquer to specify each of our individual learners and trained a meta learner to combine this experts.

**Structure**

- src: contains different scripts that compute forecast of individual and combined learners.
    - psf_clustering_script.py: script to perform the clustering then combining algorithm. It needs the result of individual learners on the whole dataset. The training_psf_dtw.py file contains useful function for the script to run
    - global_script: Script that computes the predictions of the individual learners on the whole dataset and using different clustering techniques: SOM, SOM with DTW, hourly clustering... Also computes the predictions of ensemble learners.
        - main.py: main function that takes the dataset and return predictions of individual learners as well as ensemble learners.
        - clustering.py: depending on the chosen clustering method, performs the clustering on the trained dataset
        - classification.py: train the classifier on the train dataset and predict with it the clusters on the ensemble and test datasets.
        - training.py: functions to train the models, individual learners as well as ensemble models.
        - models.py: also used by psd_clustering_script.py, definition of models and metrics used in this work.
- notebooks:
    - data_preprocessing.ipynb: notebook that presents the data prepocessing: missing values, resampling...
    - results_vizualisation.ipynb: notebook using object-oriented  to ease the vizualisation and the analyse of the data. 
