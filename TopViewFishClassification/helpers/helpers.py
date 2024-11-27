import math
import pandas as pd

### Helper functions that are used in more than one package

def pd_rolling(col, window, func, right_leaning, func_args = None):
    """
    My own implementation of pd.Series.rolling(center=True). This function does rolling window calculations.
    It applies a function to a window of data centered around a value and resets that value to the result of the 
    function. pd_rolling avoids boundary issues by repeating the boundary values to extend the boundary.
    Note: I did not use pd.Series.rolling because this function gets buggy around the boundaries if center=True, 
    so would recommend trying not to use it unless it gets updated
        
        Parameters:
            col (pandas.Series): A series of datapoints over which to do rolling window calculations
            
            window (int): size of rolling window

            func (Callable): The function to be applied to each window that determines the new value for 
            each datapoint

            right_leaning (bool): Only applies if the size of the rolling window is even. If true,
            there will be more arguments to the left of the center value than the right. If false, the
            opposite is true.

            func_args (list): list of arguments for func. Default is None

        Returns: 
            new_col (pandas.Series): The transformed series of data_points
    """
    # Add padding of repeats of boundary values. The amount of padding added to both sides
    # is currently set to half the size of the window.
    padding = math.ceil(window/2)
    new_col = pd.concat([col.iloc[[0]].repeat(padding), col], ignore_index=True)
    new_col = pd.concat([new_col, new_col.iloc[[-1]].repeat(padding)], ignore_index=True)
    
    # Create a copy of the padded series to be used for calculations so that the original
    # padded series can be used to store values
    ref_col = new_col.copy()

    # Iterate over the padded series, skipping the values that make up the padding, and apply
    # rolling window calculations
    for i in range(padding, new_col.size-padding):
        if(right_leaning):
            w = ref_col.iloc[i-math.ceil((window-1)/2):i+math.floor((window-1)/2)+1]
            if(func_args is not None):
                new_col.at[i] = func(w, func_args[i-window])
            else:
                new_col.at[i] = func(w)
        else:
            w = ref_col.iloc[i-math.floor((window-1)/2):i+math.ceil((window-1)/2)+1]
            if(func_args is not None):
                new_col.at[i] = func(w, func_args[i-window])
            else:
                new_col.at[i] = func(w)
    
    # remove padding
    new_col = new_col.tail(-padding)
    new_col = new_col.head(-padding).reset_index(drop = True)

    return new_col