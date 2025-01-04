import numpy as np

## Functions for Removing Low-Certainty Data
def interpolate_position_data(df, pdf, points, p_cut=0.6, method='spline', order=3):
    """
    For each column in a Pandas Dataframe, interpolates over low-certainty (low-likelihood) datapoints
        
        Parameters:
            df (pandas.DataFrame): Full, unaltered DLC dataframe with 'likelihood' data
            
            pdf (pandas.DataFrame): The dataframe with data to be transformed
            
            points (list): list of label names
            
            p_cut (float): The certainty value below which data points are removed. This value ranges between
            0 and 1
            
            method (string): interpolation method (from the scipy.interpolation library)
            
            order (int): order of interpolation method. Only necessary for certain methods such as 'spline' and 
            'polynomial'
        
        Returns: 
            pos_df (pandas.DataFrame): returns a new dataframe (a copy of pdf with transformations applied)
    """

    assert p_cut >= 0 and p_cut <= 1, "p_cut must be in the interval [0,1]"

    pos_df = pdf.copy()
    for point in points:
        # Extract the three columns of data for each label (x, y, likelihood)
        likelihood = df[point + ' likelihood']
        x = pos_df[point + ' x']
        y = pos_df[point + ' y']

        # Replace points with low certainty (below the p_cut) with NaNs
        x[likelihood < p_cut] = np.nan
        y[likelihood < p_cut] = np.nan

        # Apply interpolation
        x.interpolate(method=method, order=order, inplace=True)
        y.interpolate(method=method, order=order, inplace = True)
    return pos_df

def set_to_prev_point(pdf, df, skeleton, p_cut = 0.6):
    ''' 
    Sets the location of low-certainty points to the closest rostral, unobscured point on the tail.
        
        Parameters:
            df (pandas.DataFrame): Full, unaltered dataframe with 'likelihood' data
            
            pdf (pandas.DataFrame): The dataframe with data to be transformed
            
            skeleton (list): label names listed in order from the caudal-most point to the rostral-most point of 
            the tail. Low-uncertainty points will not be changed for the rostral-most point. This point will only
            be used as a reference. 
            
            p_cut (float): The certainty value below which data points are removed. This value ranges between
            0 and 1. 
        
        Returns: 
            pos_df (pandas.DataFrame): returns a new dataframe (a copy of pdf with transformations applied)
    '''
    assert p_cut >= 0 and p_cut <= 1, "p_cut must be in the interval [0,1]"

    str_err = "one or more labels listed in skeleton cannot be found in df"
    assert set([s + ' likelihood' for s in skeleton]).issubset(df.columns), str_err
    str_err = "one or more labels listed in skeleton cannot be found in pdf"
    assert set([s + ' x' for s in skeleton]).issubset(pdf.columns), str_err
    assert set([s + ' y' for s in skeleton]).issubset(pdf.columns), str_err

    pos_df = pdf.copy()

    # Find low-certainty points
    for i in range(len(skeleton)-1):
        # Extract the three columns of data for each label (x, y, likelihood)
        likelihood = df[skeleton[i] + ' likelihood']
        x = pos_df[skeleton[i] + ' x']
        y = pos_df[skeleton[i] + ' y']
        # Replace points with low certainty (below the p_cut) with NaNs
        x[likelihood < p_cut] = np.nan
        y[likelihood < p_cut] = np.nan

    # Replace low-certainty points
    for i in reversed(range(len(skeleton)-1)):
        # Extract the position data (x,y) for each label 
        x = pos_df[skeleton[i] + ' x']
        y = pos_df[skeleton[i] + ' y']
        
        # Replace NaN values with previous value
        x[x.isna().to_numpy().flatten()] = pos_df[skeleton[i+1] + ' x'].values[x.isna().to_numpy().flatten()]
        y[y.isna().to_numpy().flatten()] = pos_df[skeleton[i+1] + ' y'].values[y.isna().to_numpy().flatten()]
    return pos_df

def extend_tail(pdf, df, skeleton, p_cut = 0.6):
    ''' 
    Infers the location of low-certainty tail points by assuming low-certainty points are spaced evenly along the 
    line formed by the two closest rostral tail points.

        Parameters:
            df (pandas.DataFrame): Full, unaltered dataframe with 'likelihood' data

            pdf (pandas.DataFrame): The dataframe with data to be transformed

            skeleton (list): label names listed in order from the caudal-most point to the rostral-most point of 
            the tail. Low-uncertainty points will not be inferred for the two anterior-most points. These points will
            be used for reference only. 

            p_cut (float): The certainty value below which data points are removed. This value ranges between
            0 and 1. 

        Returns: 
            pos_df (pandas.DataFrame): the transformed pdf
    '''
    assert p_cut >= 0 and p_cut <= 1, "p_cut must be in the interval [0,1]"

    str_err = "one or more labels listed in skeleton cannot be found in df"
    assert set([s + ' likelihood' for s in skeleton]).issubset(df.columns), str_err
    str_err = "one or more labels listed in skeleton cannot be found in pdf"
    assert set([s + ' x' for s in skeleton]).issubset(pdf.columns), str_err
    assert set([s + ' y' for s in skeleton]).issubset(pdf.columns), str_err

    pos_df = pdf.copy()

    # Find low-certainty points
    for i in range(len(skeleton)-2):
        # Extract the three columns of data for each label (x, y, likelihood)
        likelihood = df[skeleton[i] + ' likelihood']
        x = pos_df[skeleton[i] + ' x']
        y = pos_df[skeleton[i] + ' y']
        # Replace points with low certainty (below the p_cut) with NaNs
        x[likelihood < p_cut] = np.nan
        y[likelihood < p_cut] = np.nan

    # Replace low-certainty points
    for i in reversed(range(len(skeleton)-2)):
        # Extract the position data (x,y) for each label 
        x = pos_df[skeleton[i] + ' x']
        y = pos_df[skeleton[i] + ' y']
        
        # Replace NaN values with an extension of the previous two points
        x_nans = x.isna().to_numpy().flatten()
        y_nans = y.isna().to_numpy().flatten()
        del_x = pos_df[skeleton[i+1] + ' x'].values[x_nans] - pos_df[skeleton[i+2] + ' x'].values[x_nans]
        del_y = pos_df[skeleton[i+1] + ' y'].values[y_nans] - pos_df[skeleton[i+2] + ' y'].values[y_nans]
        x[x_nans] = pos_df[skeleton[i+1] + ' x'].values[x_nans] + del_x
        y[x_nans] = pos_df[skeleton[i+1] + ' y'].values[y_nans] + del_y
        
    return pos_df
