import pandas as pd
import numpy as np
import math
from scipy import signal
from scipy.stats import norm
import pywt

import sys
import os

# adding helpers to the system path
package_directory = os.getcwd().split(os.pardir)[0]
sys.path.insert(0, package_directory + '/TopViewFishClassification/helpers')
from helpers import pd_rolling

## Functions for Reducing Noise

# Helper Functions
def get_gaussian(window, scale, high_p, low_p, exp):
    """
    Returns a normalized (values sum to 1) gaussian window. The standard deviation (and thus the sharpness), 
    of the gaussian is determined by the scale, high_p, low_p, and exp parameters. The high_p and low_p 
    values define a range of possible std values. The low_p value returns a lower std bound defined as the 
    std at which the interval [center_of_window - 0.5, center_of_window + 0.5] contains low-p area under the 
    gaussian curve. The high_p value returns an upper std bound defined as the std at which the entire window 
    contains high_p area under the gaussian curve. The scale parameter then determines where along this std
    interval, determined by the upper and lower std bounds, the std of the returned gaussian window will lie.
    The scale parameter is a value between 0 and 1, with 1 being equal to the lower bound std and 0 to the 
    upper bound std. The std mapping from the scale value between 0 and 1 is determined by the 
    exp parameter. If exp = 1, uniformally spaced scale values correspond to uniformally spaced points along 
    the std interval. If exp > 1, uniformally spaced scale values correspond to exponentially spaced points
    of degree=exp along the std interval. The exp parameter is meant to control how quickly the gaussian window 
    sharpens and flattens with changing scale values. If exp>>1, only very high scale values will result in a
    sharp gaussian curve that does not perform like a boxcar filter.
        
        Parameters:
            window (int): Size of gaussian window to be returned

            scale (float): A value between 0 and 1 that maps to the std of the window to be returned according to the 
            mapping determined by the values high_p, low_p, and exp. 

            high_p (float): Determines the upper std bound of the std mapping defined as the std at which the entire 
            window contains high_p area under the gaussian curve. Must be between 0 and 1.

            low_p (float): Determines the lower bound of the std interval defined as the std at which the interval 
            [center_of_window - 0.5, center_of_window + 0.5] contains low-p area under the gaussian curve. Must be
            between 0 and 1.

            exp (int, float): Determines the degree of the exponential mapping from the scale interval [0,1] to the
            std interval determined by high_p and low_p. 

        Returns: 
            gauss (ndarray): Normalized (values sum to 1) gaussian window
    """
    
    str_err = " must be in the interval [0,1]"
    assert scale >= 0 and scale <= 1, "scale" + str_err
    assert high_p >= 0 and high_p <= 1, "high_p" + str_err
    assert low_p >= 0 and low_p <= 1, "low_p" + str_err

    # Applies exponential mapping. If exp = 1, no change is made
    scale = (1 - scale) ** exp
    
    # Determines the bounds of the std interval
    high_std = (window-1)/(2*norm.ppf((high_p + 1)/2))
    low_std = 1/(2*norm.ppf((low_p + 1)/2))

    # Gets std value from scale value
    std = scale*high_std + (1-scale)*low_std

    # Generates gaussian window
    gauss = signal.windows.gaussian(window, std)

    # Normalizes gaussian window
    gauss = gauss/gauss.sum()

    return gauss

def apply_gaussian(window, scale, high_p, low_p, exp):
    """
    Applies a normalized gaussian filter to the provided window. The gaussian filter is generated based off of the scale, 
    high_p, low_p, and exp parameters. See get_gaussian for more details on how the gaussian window is generated.

        Parameters:
            window (int): Size of gaussian window to be returned

            scale (float): A value between 0 and 1 that maps to the std of the gaussian window according to the 
            mapping determined by the values high_p, low_p, and exp. Must be between 0 and 1

            high_p (float): Determines the upper bound of std mapping defined as the std at which the entire 
            gaussian window contains high_p area under the gaussian curve. Must be between 0 and 1. Same
            parameter as in get_gaussian

            low_p (float): Determines the lower bound of the std mapping defined as the std at which the interval 
            [center_of_window - 0.5, center_of_window + 0.5] of the gaussian window contains low-p area under the 
            gaussian curve. Must be between 0 and 1. Same parameter as in get_gaussian

            exp (int, float): Determines the degree of the exponential mapping from the scale interval [0,1] to the
            std interval determined by high_p and low_p. Same parameter as in get_gaussian

        Returns: 
            new_window (ndarray): the transformed window with gaussian filter applied
    """
    # Get gaussian filter
    gauss = get_gaussian(len(window), scale, high_p, low_p, exp)

    # Apply gaussian filter
    new_window = np.multiply(gauss, window.copy().to_numpy()).sum()
    
    return new_window

def avoid_missing_data(window, func, right_leaning):
    """
    This function acts as a wrapper to functions passed into pd_rolling. This function slices a window containing NaN 
    values into the largest possible window of consecutive non-NaN values containing the center value of the window.
    (Note that the center value is the value to be modified in pd_rolling)

        Parameters:
            window (pd.Series): window extracted by pd_rolling

            func (Callable): function to be applied to window

            right_leaning (bool): Same parameter as in pd_rolling. Only applies if the size of the rolling window is even. 
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

        Returns: 
            The new value for the center value of the window
    """
    # Convert window from pd.Series to 1-d numpy array
    window = window.copy().to_numpy().flatten()

    # If there are no NaNs, keep original window. Otherwise slice window
    if np.isnan(window).any():
        
        # Determine the center value of the window
        if right_leaning:
            center = math.ceil((len(window)-1)/2)
        else:
            center = math.floor((len(window)-1)/2)

        # If the center value is NaN, return NaN. The idea is to ignore
        # NaN values, so there is no reason to change a NaN value
        if np.isnan(window[center]):
            return np.nan
        
        # Find where NaN values are, and then slice window around the 
        # central value to exclude NaN values
        nans = np.argwhere(np.isnan(window)).flatten()
        left = 0
        right = len(window)
        if (nans < center).any():
            left = np.max(nans[nans < center]) + 1
        if (nans > center).any():
            right = np.min(nans[nans > center])
        window = window[left:right]
    
    # Apply the filter function to the new window and return the new center value
    return func(window)

def correlation(col1, col2, ind_norm, inverse, zero_min):
    """
    Calculates the correlation between two pandas DataFrame columns. In this case, the correlation is defined
    as the element-wise product of the two normalized columns.

        Parameters:
            col1 (pandas.Series): The first column

            col2 (pandas.Series): The second column

            ind_norm (bool): If true, normalize each column individually. Otherwise normalize them together. 
            To normalize a column, the mean is subtracted from the column to center the data around 0 and 
            then the column is divided by the maximum value to bring the range of the data to [-1,1]. 
            If columns are normalized together they are both divided by the maximum value of both columns
            so that the columns retain their relative values.

            inverse (bool): If true, multiplies the cross-correlation by -1. Otherwise calculates the normal
            cross-correlation.

            zero_min (bool): If true, change the correllation values to fit within the interval [0,1] with the 
            min value corresponding to 0 and the max value corresponding to 1

        Returns: 
            cross-corr (pandas.Series): The correlation by value of the two inputted dataframe columnns.
    """
    # Get copies of columns to apply transformation to
    c1 = col1.copy()
    c2 = col2.copy()

    # Recenter columns around 0
    c1 = c1 - np.mean(c1)
    c2 = c2 - np.mean(c2)

    c1_max = np.max(np.abs(c1))
    c2_max = np.max(np.abs(c2))

    # Rescale columns to values between [-1,1]
    if(ind_norm):
        # If ind_norm is True, divide each column
        # by its own maximum values
        c1 = c1 / c1_max
        c2 = c2 / c2_max
    else:
        # If ind_norm is False, divide each
        # column by the maximum of both columns
        c1 = c1 / np.max([c1_max,c2_max])
        c2 = c2 / np.max([c1_max,c2_max])

    # Reverse sign of correlation array if inverse is true
    if(inverse):
        c1 = -1 * c1
    
    # Calculate the element-wise correlation
    corr = c1.mul(c2)

    # If zero_min is true, rescale the correlation vector
    # to fit within the interval [0,1]
    if(zero_min):
        corr = corr - np.min(corr) 
        corr = corr / np.max(corr) 

    return corr


# Filter Functions
def median_filter(df, window, right_leaning=True):
    """
    Applies the median filter to each column of a pandas DataFrame. The median filter replaces each value with
    the median of the surrounding window of values in the same column.

        Parameters:
            df (pandas.DataFrame): The dataframe on which to apply the transformation

            window (list[int,...]): The size of the window to be applied to each DataFrame column. If don't want to apply filter
            to column, set window size to 0

            right_leaning (bool): Same parameter as in pd_rolling. Only applies if the size of the rolling window is even. 
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

        Returns: 
            The modified DataFrame
    """
    # Make copy of dataframe on which to apply transformations
    new_df = df.copy()

    # Iterate over columns of dataframe, applying mean filter to each column
    for i in range(len(df.columns)):
        
        # Only Apply filter if window size is greater than zero
        if window[i] > 0:
            new_df[df.columns[i]] = pd_rolling(df[df.columns[i]], window[i], np.median, right_leaning)
        
    return new_df 

def mean_filter(df, window, right_leaning=True):
    """
    Applies mean filter to each column of a pandas DataFrame. The mean filter replaces each value with
    the mean of the surrounding window of values in the same column.

        Parameters:
            df (pandas.DataFrame): The dataframe on which to apply transformation

            window (list[int,...]): size of window to be applied to each DataFrame column. If don't want to apply filter
            to column, set window size to 0

            right_leaning (bool): Same parameter as in pd_rolling. Only applies if the size of the rolling window is even. 
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

        Returns: 
            (pandas.DataFrame): The modified DataFrame
    """
    # Make copy of dataframe on which to apply transformations
    new_df = df.copy()

    # Iterate over columns of dataframe, applying mean filter to each column
    for i in range(len(df.columns)):
        
        # Only Apply filter if window size is greater than zero
        if window[i] > 0:        
            new_df[df.columns[i]] = pd_rolling(df[df.columns[i]], window[i], np.mean, right_leaning)

    return new_df 

def adaptive_gaussian_filter(df, window, order=1, right_leaning=True, high_p=0.2, low_p=0.99, exp=15):
    """
    Applies a gaussian filter to each column of a pandas DataFrame. The std of the filter for each window
    is determined by the absolute magnitude of the nth order derivative over the center value of the window.
    The 'absolute magnitude of the the nth order derivative over the center value' is calculated by summing 
    together the absolute values of the nth derivatives from the center value and the previous value and the 
    center value and the next value.

        Parameters:
            df (pandas.DataFrame): The dataframe on which to apply the transformation

            window (list[int,...]): size of window to be applied to each DataFrame column. If don't want to apply filter
            to column set window size to 0

            order (int): Indicates which derivative will be used to determine the std of each window.
            This includes the absolute amplitude (order = 0), velocity (order = 1), acceleration 
            (order = 2) etc. over the center value.

            right_leaning: Same parameter as in pd_rolling. Only applies if the size of the rolling window is even.
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

            high_p (float): Determines the upper bound of the std mapping defined as the std at which the entire 
            gaussian window contains high_p area under the gaussian curve. Must be between 0 and 1. Same
            parameter as in get_gaussian.

            low_p (float): Determines the lower bound of the std mapping defined as the std at which the interval 
            [center_of_window - 0.5, center_of_window + 0.5] of the gaussian window contains low-p area under the 
            gaussian curve. Must be between 0 and 1. Same parameter as in get_gaussian

            exp (int, float): Determines the degree of the exponential mapping from the scale interval [0,1] to the
            std interval determined by high_p and low_p. Same parameter as in get_gaussian

        Returns: 
            new_window (ndarray): the transformed window with gaussian filter applied
    """
    assert order >= 0, "order must be a positive integer"

    # Make copy of dataframe on which to apply transformations
    new_df = df.copy()

    # Iterate over columns of dataframe
    for i in range(len(df.columns)):
        # Only apply filter if window size is greater than zero
        if window[i] > 0:
            col = new_df[df.columns[i]]

            # If order > 0, gets difference between adjacent points on both the
            # the points before and after each value until get desired order
            # of derivate 
            if order > 0:
                left_diff = col
                right_diff = col.copy()
                for j in range(order):
                    left_diff = left_diff.diff(periods=1).fillna(0)
                    right_diff = right_diff.diff(periods=-1).fillna(0)
                # Returns a sum of the absolute values of the calculated nth derivatives
                # between the center values and the previous and next values
                scales = np.add(np.abs(right_diff), np.abs(left_diff))
            
            # If order == 0 just returns absolute amplitude of values
            else:
                scales = np.abs(col)
            
            # Converts whatever property is being used into a scaling factor
            # between 0 and 1 for each value in the column
            scales = (scales/np.max(scales)).to_numpy().flatten()

            # Applies gaussian to each window with std set according to the "scale" of property of interest of the corresponding center value
            new_df[df.columns[i]] = pd_rolling(col, window[i], (lambda w, s: apply_gaussian(w, s, high_p, low_p, exp)), right_leaning = right_leaning, func_args=scales)   
    return new_df

def threshold_filter(df, window, thresh, func, order=1, right_leaning = True):
    """
    Applies a filter (specified by func) to each column of a pandas DataFrame while ignoring values
    with an absolute magnitude of the nth order derivative over the value above a certain threshold.
    The 'absolute magnitued of the the nth order derivative over the value' is calculated by summing 
    together the absolute values of the nth derivatives between the value and the previous value and the 
    value and the next value.

        Parameters:
            df (pandas.DataFrame): The dataframe on which to apply the transformation

            window (list[int,...]): size of window to be applied to each DataFrame column. If don't want to apply filter
            to column set window size to 0

            thresh (list[int,...]): The threshold values to be used for each column. Should be between
            0 and 1. The threshold value has the same units as the normalized column (the column subtracted
            by the mean and divided by the max value.).

            func (Callable): The filtering method (or function to be applied to each rolling window)

            order (int): Indicates which derivative will be compared to the threshold value.
            This includes the absolute amplitude (order = 0), velocity (order = 1), acceleration 
            (order = 2) etc. over the center value.

            right_leaning (bool): Same parameter as in pd_rolling. Only applies if the size of the rolling window is even.
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

        Returns: 
            new_window (ndarray): the transformed window with threshold filter applied
    """
    assert order >= 0, "order must be a positive integer"

    # Make copy of dataframe on which to apply transformations
    new_df = df.copy()

    # Iterate over columns of dataframe
    for i in range(len(df.columns)):
        assert thresh[i] >= 0 and thresh[i] <= 1, "all thresh values must be in the interval [0,1]"

        # Only apply filter if window size is greater than zero
        if window[i] > 0:
        
            # Normalize dataframe column (by subtracting mean and dividing by max value)
            norm = new_df[df.columns[i]] - np.mean(new_df[df.columns[i]])
            norm = norm / np.max(np.abs(norm))

            # If order > 0, gets difference between adjacent points on both the
            # the points before and after each value until get desired order
            # of derivate 
            if order > 0:
                prev_diff = norm
                next_diff = norm.copy()
                for j in range(order):
                    prev_diff = prev_diff.diff(periods=1).fillna(0)
                    next_diff = next_diff.diff(periods=-1).fillna(0)
                
                # Normalize to 0 to 1 range
                prev_diff = np.abs(prev_diff)
                next_diff = np.abs(next_diff)
                prev_diff = prev_diff / (np.max(prev_diff))
                next_diff = next_diff / (np.max(next_diff))

                # Set values with at least one derivative above the threshold to NaN
                new_df[df.columns[i]].values[np.abs(prev_diff) > thresh[i]] = np.nan
                new_df[df.columns[i]].values[np.abs(next_diff) > thresh[i]] = np.nan

            # If order == 0, use absolute amplitude as porperty to determine which
            # values to ignore during fitlering
            else:
                # Set values with absolute values above the threshold to NaN
                new_df[df.columns[i]].values[np.abs(norm) > thresh[i]] = np.nan
            
            # Apply filter. Windows will be sliced to exclude NaN values during filtering
            new_df[df.columns[i]] = pd_rolling(new_df[df.columns[i]], window[i], (lambda w: avoid_missing_data(w, func, right_leaning)), right_leaning = right_leaning)

            # Recover NaN values from original column
            new_df[df.columns[i]].values[new_df[df.columns[i]].isna().to_numpy().flatten()] = df[df.columns[i]].values[new_df[df.columns[i]].isna().to_numpy().flatten()]

    return new_df

def threshold_match_filter(col1_x, col1_y, col2_x, col2_y, window, thresh, func, ind_norm = False, inverse = True, right_leaning = True):
    """
    Applies a filter (specified by func) to four pandas.Series columns (the x and y values of two labels). The filter 
    ignores values that have a 'correlation' above the threshold value. The correlation value is determined by the 
    product of the normalized elementwise products of the two columns of x values and the two columns of y values.
    This is meant to act as way to filter out noise from labels trajectories for labels that are supposed to move
    simultaneously.

        Parameters:
            col1_x (pandas.Series): First label x values over time

            col1_y (pandas.Series): First label y values over time

            col2_x (pandas.Series): Second label x values over time

            col2_y (pandas.Series): Second label y values over time         
            
            window (list[int,int]): size of window to be applied to each column. If don't want to apply filter
            to column set window size to 0

            thresh (list[int,int]): The threshold values to be used for each column. Should be between
            0 and 1. This value will be compared to the normalized correlation between the two labels
            at each time point

            func (Callable): The filtering method (or function to be applied to each rolling window)

            ind_norm (bool): Same parameter as in correlation. If true, normalize each column individually. Otherwise 
            normalize them together. To normalize a column, the mean is subtracted from the column to center 
            the data around 0 and then the column is divided by the maximum value to bring the range of the 
            data to [-1,1]. If columns are normalized together they are both divided by the maximum value of 
            both columns so that the columns retain their relative values.

            inverse (bool): Same parameter as in correlation. If true, multiplies the cross-correlation by -1. Otherwise 
            calculates the normal cross-correlation. This is to be applied if the labels should move simulatenously but in 
            opposite directions rather than the same direction

            right_leaning (bool): Same parameter as in pd_rolling. Only relevant if the size of the rolling window is even.
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

        Returns: 
            new_window (ndarray): the transformed window with threshold filter applied
    """
    # Calculate element-wise correlation
    corr_x = correlation(col1_x, col2_x, ind_norm, inverse, True)
    corr_y = correlation(col1_y, col2_y, ind_norm, inverse, True)
    corr = np.multiply(corr_x, corr_y)
    corr = corr / np.max(corr)

    # Get copies of columns to which to apply transformations
    cols = (col1_x, col1_y, col2_x, col2_y)
    new_cols = [col1_x.copy(), col1_y.copy(), col2_x.copy(), col2_y.copy()]

    for i in range(len(new_cols)):
        assert thresh[i] >= 0 and thresh[i] <= 1, "all thresh values must be in the interval [0,1]"

        # Only apply filter if window size is greater than zero
        if window[i] > 0:
            
            # Set values with correlations above the threshold to NaN
            new_cols[i][corr > thresh[i]] = np.nan
            
            # Apply filter. Windows will be sliced to exclude NaN values during filtering
            new_cols[i] = pd_rolling(new_cols[i], window[i], (lambda w: avoid_missing_data(w, func, right_leaning)), right_leaning = right_leaning)
            
            # Recover NaN values from original columns
            new_cols[i][new_cols[i].isna().to_numpy().flatten()] = cols[i][new_cols[i].isna().to_numpy().flatten()]
    
    return new_cols

def gaussian_match_filter(col1_x, col1_y, col2_x, col2_y, window, ind_norm = False, inverse = True, right_leaning = True, high_p=0.2, low_p=0.99, exp=15):
    """
    Applies a gaussian filter (specified by func) to four pandas.Series columns (the x and y values of two labels). 
    The std of the filter for each window is determined by the correlation strength between the two labels at the
    center value of the window. The correlation value is determined by the product of the normalized elementwise products of 
    the two columns of x values and the two columns of y values. This is meant to act as way to filter out noise 
    from labels trajectories for labels that are supposed to move simultaneously.

        Parameters:
            col1_x (pandas.Series): First label x values over time

            col1_y (pandas.Series): First label y values over time

            col2_x (pandas.Series): Second label x values over time

            col2_y (pandas.Series): Second label y values over time         
            
            window (list[int,int]): size of window to be applied to each column. If don't want to apply filter
            to column set window size to 0
            
            ind_norm (bool): Same parameter as in correlation. If true, normalize each column individually. Otherwise 
            normalize them together. To normalize a column, the mean is subtracted from the column to center 
            the data around 0 and then the column is divided by the maximum value to bring the range of the 
            data to [-1,1]. If columns are normalized together they are both divided by the maximum value of 
            both columns so that the columns retain their relative values.

            inverse (bool): Same parameter as in correlation. If true, multiplies the cross-correlation by -1. Otherwise 
            calculates the normal cross-correlation. This is to be applied if the labels should move simulatenously but in 
            opposite directions rather than the same direction

            right_leaning: Same parameter as in pd_rolling. Only applies if the size of the rolling window is even.
            If true, there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

            high_p (float): Determines the upper bound of the std interval defined as the std at which the entire 
            gaussian window contains high_p area under the gaussian curve. Must be between 0 and 1. Same
            parameter as in get_gaussian.

            low_p (float): Determines the lower bound of the std interval defined as the std at which the interval 
            [center_of_window - 0.5, center_of_window + 0.5] of the gaussian window contains low-p area under the 
            gaussian curve. Must be between 0 and 1. Same parameter as in get_gaussian

            exp (int, float): Determines the degree of the exponential mapping from the scale interval [0,1] to the
            std interval determined by high_p and low_p. Same parameter as in get_gaussian

        Returns: 
            new_window (ndarray): the transformed window with gaussian filter applied
    """
    
    # Calculate element-wise correlation
    corr_x = correlation(col1_x, col2_x, ind_norm, inverse, True)
    corr_y = correlation(col1_y, col2_y, ind_norm, inverse, True)
    corr = np.multiply(corr_x, corr_y)
    corr = corr / np.max(corr)

    # Get copies of columns to which to apply transformations
    new_cols = [col1_x.copy(), col1_y.copy(), col2_x.copy(), col2_y.copy()]

    for i in range(len(new_cols)):
        # Only apply filter if window size is greater than zero
        if window[i] > 0:
            # Applies a gaussian with std based off of the 'scale' or stength of correlation of each center value to each column
            new_cols[i] = pd_rolling(new_cols[i], window[0], (lambda w, s: apply_gaussian(w, s, high_p, low_p, exp)), right_leaning = right_leaning, func_args=corr)
    
    return new_cols

def wavelet_threshold(df, apply, filter_thresh, freq_frac = 0.5, wavelet = 'sym6', restore = True):
    """
    Applies wavelet threshold filter to each column of a Pandas dataframe. This filter works by using the 
    wavelet transform to perform multiresolution analysis on each label trajectory. Multiresolution 
    analysis decomposes each trajecotry into time-localized frequency components. The threshold
    filter is only applied to the higher frequency components, since noise is most often
    high-frequency. In these highest frequncy components, the values less than the threshold
    distance away from the mean are reduced to the mean. The components are then summed
    back together to recover the filtered label trajectory. This method will often
    still filter over values that are important to the signal, so there is an option
    to restore values that are above the threshold distance away from the mean to their orignal 
    value.

    Parameters:
            df (pandas.DataFrame): The dataframe on which to apply transformation
            
            apply (list[bool,...]): Contains boolean values that indicate whether or not to apply the filter
            to each column
            
            filter_thresh (list[float,...]): For each column, the distance away from the mean at which the 
            high-frequency component values are reduced to the mean. For each column, the threshold value is 
            in units of the standard deviation of the label trajectory.

            freq_fraq (float): The fraction of the frequency components to which to apply thte filter
            
            wavelet (string): The type of wavelet to be used for multiresolution analysis. For options see:
            https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html

            restore (bool); Whether or not to restore to their original value the filtered values greater than 
            the threshold distance away from the mean

        Returns: 
            new_df (pd.DataFrame): The transformed dataframe
        """

    # Make copy of dataframe on which to apply transformations
    new_df = pd.DataFrame()

    # Iterate over columns of dataframe, applying mean filter to each column
    for i in range(len(df.columns)):
        # Only Apply filter if window size is greater than zero
        if apply[i]:       
            # Get column as ndarray
            col = df[df.columns[i]].copy()
            
            # Apply Normalization
            col_mean = np.mean(col)
            col = col - col_mean
            col_max = np.max(np.abs(col))
            col = col / col_max

            # Save the original trajectory values
            restore_thresh = filter_thresh[i]

            # Apply Multiresolution Analysis
            sigma = np.std(col)
            mra = pywt.mra(col, wavelet, None, axis=0, transform='dwt')

            # Apply a threshold filter to the highest frequencies 
            for j in range(math.ceil(len(mra)*freq_frac), len(mra)):
                mra[j] = pywt.threshold(mra[j], sigma*filter_thresh[i], mode='hard') 
            
            # Recover Filtered Data
            filtered = np.sum(mra, axis=0)

            # Restore large values that should not have been filtered if restore is true
            if restore:
                filtered[np.abs(col) > sigma*restore_thresh] = col[np.abs(col) > sigma*restore_thresh]

            # Undo Normalization
            new_df[df.columns[i]] = (filtered * col_max) + col_mean
        else:
            new_df[df.columns[i]] = df[df.columns[i]]

    return new_df 
