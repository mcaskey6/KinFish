import numpy as np
import math
import pandas as pd
from dtaidistance import dtw_ndim

import sys
import os

# adding helpers to the system path
package_directory = os.getcwd().split(os.pardir)[0]
sys.path.insert(0, package_directory + '/TopViewFishClassification/helpers')
from helpers import pd_rolling

## Functions for compiling test and training data from an array of dataframes

def compile_tree_data(dfs, train_split = 0.95, fish_split = None, active_rest_split = None, left_window = 15, right_window = 5):
    """
    Given a list of dataframes containing the extracted features from a series of recordings, this function separates
    the data into a training set and a test set. It then arranges the data into a format which can be inputted
    into an sklearn classifier. In order to classify a frame as active or inactive, the extracted features from
    that frame and a window of surrounding frames are inputted into the classifier.

    Parameters:
        dfs (list[pandas.DataFrame,...]): list of DataFrames containing the extracted features from each recording

        train_split (float): ratio of training to test data

        fish_split (list[float,...]): List containing the ratio of training data that should be extracted from each 
        recording. Should add up to 1. This list should be the same length as dfs. If None, will extract an 
        equal ratio of training data from each recording

        active_rest_split (float): the fraction of the training dataset which should be active frames. If None,
        randomly sample all data

        left_window (int): The number of preceding frames whose extracted features will be included in the
        input to the classifier to decide whether a certain frame is exhibiting activity or rest

        right_window (int): The number of subsequent frames whose extracted features will be included in the
        input to the classifier to decide whether a certain frame is exhibiting activity or rest

    Returns:
        all_data (list[ndarray,...]): List of the formatted input data for all frames in each recordings
        x_train (ndarray): Formatted data for training frames
        y_train (ndarray): Ground truth value for training frames
        x_test (ndarray): Formatted data for test frames
        y_test (ndarray): Ground truth values for test frames 
        x (ndarray): Formatted data for all frames (across all recordings)
        y (ndarray): Grouund truth values for all manually labelled frames
    """
    assert train_split >= 0 and train_split <= 1, "train_split must be in the interval [0,1]"

    # Extract an equal ratio of training data from each recording if fish_split is not specified
    labelled_nums = [len(df[df['Active']!=-1]) for df in dfs]
    if fish_split is None:
         fish_split = labelled_nums / np.sum(labelled_nums)
    else:
        assert sum(fish_split) == 1, "fish_split must sum to 1"
        assert len(fish_split) == len(dfs), "fish_split must be the same length as dfs"

    all_data = []
    all_train_ind = []
    all_test_ind = []

    # Iterate through each recording
    for i, df in enumerate(dfs):
        set_size = math.floor(np.sum(labelled_nums) * train_split * fish_split[i])
        # Discard the unlabelled frames
        tree_df_labelled = df[df['Active'] != -1]

        if active_rest_split is not None:
            # How much of the training data from a single recording should be of active frames
            active_size = math.ceil(set_size * active_rest_split)
            active_frames = sum(tree_df_labelled['Active']==1)
            print(active_frames)
            print(active_size)
            assert active_size <= active_frames, ("The active_rest_split is too large given the number" 
                                                  " of active frames in the fish recording at index " + str(i) + 
                                                  ". Given the train_split and fish_split, the number of active frames" 
                                                  " taken from this fish recording for training would be " + 
                                                  str(active_size) + " but there are only "
                                                  + str(active_frames) + " active frames in this recording.")
            # How much of the training data from a single recording should be of rest frames
            rest_size = math.floor(set_size * (1 - active_rest_split))
            rest_frames = sum(tree_df_labelled['Active']==0)
            assert rest_size <= rest_frames, ("The active_rest_split is too small given the number" 
                                                " of rest frames in the fish recording at index " + str(i) + 
                                                ". Given the train_split and fish_split, the number of rest frames" 
                                                " taken from this fish recording for training would be " + 
                                                str(rest_size) + " but there are only "
                                                + str(rest_frames) + " rest frames in this recording.")

        assert set_size <= len(tree_df_labelled), ("The fish split value of " + str(fish_split[i]) + " at index "
                                                   + str(i) + " is too large. Given the train_split, " 
                                                   "the number of training frames taken from this fish recording" 
                                                   " would be " + str(set_size) + " but there are only "
                                                   + str(len(tree_df_labelled)) + " frames in this recording.")

        # If specified, control the ratio of active to quiescent frames in the dataset
        if active_rest_split is not None:
            # Randomly sample active frames for the training set
            active = np.random.choice(tree_df_labelled.index[tree_df_labelled['Active'] == 1].tolist(), size=active_size, replace=False)
            # Randomly sample inactive frames for the training set
            inactive = np.random.choice(tree_df_labelled.index[tree_df_labelled['Active'] == 0].tolist(), size=rest_size, replace=False)
            # Collect all training data for the recording into a single array
            train_ind = np.sort(np.append(active, inactive))
        else:
            train_ind = np.sort(np.random.choice(tree_df_labelled.index[:].tolist(), size=set_size, replace=False))

        # Store the training data indices for the recording in a list which stores the training indices for all recordings
        all_train_ind.append(train_ind)
        # Store the test data indices for the recording in a list which stores the training indices for all recordings
        test_ind = np.sort(tree_df_labelled.index[~tree_df_labelled.index.isin(train_ind)].to_numpy())
        all_test_ind.append(test_ind)
        
        # Format the data for training by iterating through each column (except the final column which contains the manual labels)
        for i in range(len(df.columns)-1):
            col = df[df.columns[i]]
            # For each frame, both the extracted features from that frame and a window of surrounding frames will be inputted
            # into the classifier
            for j in range(-right_window, left_window+1):
                    if j < 0:
                            fill = col.iloc[-1]
                    else:
                            fill = col.iloc[0]
                    shift_col = col.shift(j).fillna(fill).to_numpy().reshape(-1,1)
                    if i == 0 and j==-right_window:
                            data = shift_col
                    else:
                            data = np.hstack([data, shift_col])
        # Save the formatted data for each recording in a list which stores the formatted data for all recordings
        all_data.append(data)

    # Iterate through the data for each recording
    for i, (df, data, train_ind, test_ind) in enumerate(zip(dfs, all_data, all_train_ind, all_test_ind)):

        # Separate the formatted data into separate arrays with the training data, test data, and labelled data. 
        # Also separate the active/rest labels into training, test, and all labelled data arrays
        xtr = data[train_ind, :]
        ytr = df['Active'].values[train_ind]
        xt = data[test_ind, :]
        yt = df['Active'].values[test_ind]
        xi = data[np.concatenate([test_ind, train_ind])]
        yi = df['Active'].values[np.concatenate([test_ind, train_ind])]

        # Combine the data from all the recordings into a single array
        if i == 0:
            x_train = xtr
            y_train = ytr
            x_test = xt
            y_test = yt
            x = xi
            y = yi
        else:
            x_train = np.vstack([x_train, xtr])
            y_train = np.concatenate([y_train, ytr])
            x_test = np.vstack([x_test, xt])
            y_test = np.concatenate([y_test, yt])
            x = np.vstack([x, xi])
            y = np.concatenate([y, yi])

    return all_data, x_train, y_train, x_test, y_test, x, y

def get_active_intervals(datas, clf, median_filter_window = 0):
    """
    Groups consecutive frames of activity into intervals of activity. A single interval is defined by the frame number 
    for the start of the interval, the frame number for the end of the interval, and the recording from which the 
    interval originates. If median_filter_window > 0, the function will filter out intervals of activity or rest 
    smaller than floor(median_filter_window/2)

    Parameters: 
        datas (ndarray): A list of the formatted data for each recording to be inputted into the sklearn classifier

        clf: Trained sklearn classifier

        median_filter_window (int): If median_filter_window > 0, the function will filter out intervals of activity 
        or rest smaller than floor(median_filter_window/2)

    Returns:
        intervals (ndarray): Array specifying each interval of activity. Each row of the array details a single 
        interval of activity according to the frame number for the start of the interval, the frame number for the
        end of the interval, and the recording from which the interval originates

    """
    intervals = [0, 0, 0]

    # Iterate through the data for each recording
    for i, data in enumerate(datas):
        pred_i = clf.predict(data)
        
        # Filter out intervals of activity less than a certain value
        if median_filter_window > 0:
            pred_i = pd_rolling(pd.Series(data=pred_i), median_filter_window, np.median, right_leaning=True).to_numpy()
            pred_i = pd_rolling(pd.Series(data=pred_i), median_filter_window, np.median, right_leaning=True).to_numpy()
        
        interval = [0, 0, i]
        last_was_one = False
        # Iterate through each frame in the recording
        for j in range(len(pred_i)):
            if pred_i[j] == 1:
                # If the previous frame was not active, but this was is,
                # make this frame the beginning of an active frame
                if not last_was_one:
                    interval[0] = j
                    last_was_one = True
                else: 
                    # If get to the end of the recording, end 
                    # any interval of activity
                    if j == len(pred_i)-1:
                        interval[1] = len(pred_i)
                        intervals = np.vstack([intervals, interval])
            else:
                # If the previous frame was active, but this one is not,
                # mark this frame as the end of the interval
                if last_was_one:
                    interval[1] = j
                    intervals = np.vstack([intervals, interval])
                    last_was_one = False
    return intervals[1:, :]

def generate_correlation_matrix(intervals, dfs):
    """
    Calculates and stores the dynamic time warping distance between each pairing of intervals.
    
    Parameters:
        intervals (ndarray): array specifying the location of each interval of activity. This is
        an n x 3 array. The first column specifies the frame number for the start of each interval;
        the second column specifies the frame number for the end of each interval; and the third
        column specifies from which recording the interval originates
        
        dfs (list(pd.DataFrame)): A list of the extracted features from each recording

    Returns: (ndarray) A matrix of the dynamic time warping distances between each pair of 
    intervals
    """
    
    series = []

    # Iterate through each interval
    for interval in intervals:
        df = dfs[interval[2]]
        # Compile the extracted features from each interval into a single array
        angles = df.values[interval[0]:interval[1], :-1]
        # Add each array of extracted features to a list that contains the 
        # extracted features for all intervals of activity
        series.append(angles)
    
    # Calculate the dynamic time warping distance between each interval
    return dtw_ndim.distance_matrix(series)