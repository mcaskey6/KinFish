# KinFish
A semi-supervised machine learning pipeline for identifying and extracting swim patterns from head-fixed zebrafish video tracking experiments. 

An example analysis of a set of four recordings is outlined in the DLC_shuffle1 folder. For each recording to be analyzed, the code takes csv files containing bodypart tracking data generated by the DeepLabCut software for that recording. The data for each recording are individually processed in Jupyter notebooks named after that recording. The processed data from all four recordings is then analyzed in behavioral_classification.ipynb, which identifies and extracts the zebrafish swim patterns. Note that the Jupyter notebooks expect .avi videos of the original recordings for visualization, but these videos have not been included in this repository. 

The TopViewFishClassification folder contains two packages containing all of the functions used in the pipeline: DLC_processing for data processing of each recording and tail_classification for the combined analysis. The environment.yaml file specifies the conda enviroment. 

The results are discussed in the following paper: https://docs.google.com/document/d/1g2XskAzIeHTrLvvUlRImpuT9yDgvG9xNc57OYNPnOEo/edit?usp=sharing
