import pandas as pd
import numpy as np

def process_csv(csv_file):
    """
    Transforms a DLC csv file into a Pandas dataframe that can be used for further analysis by getting rid 
    of excess/redundant columns and rows that are part of the original file. This function also returns a list
    of the label names used for identifying different parts of the zebrafish.
        
        Parameters:
            csv_file (string): the filepath to the DLC csv file with zebrafish pose data
        
        Returns: 
            df (pandas.DataFrame): dataframe for further analysis
            
            points (list[string,...]): list of label names
    """
    df = pd.read_csv(csv_file)
    
    # Drops the "scorer column" which just gives the indices of the frames
    df = df.drop(columns=['scorer'])
    
    points = []
    
    # Rename columns based on the information from the first three rows of the csv. These rows together contain 
    # all of the info about each column. In order to perform further analysis, these rows must be consolidated
    # into a single list of column names.
    for i in range(len(df.columns)):
        old_columns = df.columns
        new_column  = df.iloc[0,i] + ' ' + df.iloc[1,i]
        points.append(df.iloc[0,i])
        df = df.rename(columns={old_columns[i]:new_column})
    df = df.drop([0,1])
    
    # Ensures the type of each column is a float and resets the indices to match the frame number
    df = df.astype('float')
    df = df.reset_index(drop = True)
   
    # Gets rid of duplicates in the label names list
    points = list(set(points))
    
    return df, points

def concatenate_clips(csv_files):
    """
    Combines the csv files for each of the video clips that make up the larger
    recording.

    Parameters:
        csv_files (list[string,...]): List of paths to each of the video clips

    Returns:
        df (Pandas.dataframe): Dataframe with the pose data for the entire 
        recording
        points (list[string,...]): List with the names for each bodypart label
    """

    # Iterate through each clip
    for i, csv_file in enumerate(csv_files):
        
        # Load clip into a Pandas DataFrame Format
        clip_df, points = process_csv(csv_file)
        # Label which video clip each frame came from
        clip_df['Clip'] = (np.ones(clip_df.shape[0])*i).astype(int)

        # Concatenate DataFrame Together
        if i==0:
            df = clip_df
        else:
            df = pd.concat([df, clip_df], axis = 0)

        df = df.reset_index(drop = True)
    return df, points

def extract_position_data(df):
    """
    Generates a new dataframe from the original dataframe without "likelihood" columns. This new dataframe will
    only contain position data.
        
        Parameters:
            df (pandas.DataFrame): Original dataframe with "likelihood" data
        
        Returns: 
            new_df (pandas.DataFrame): Dataframe containing only position data
    """
    new_df = df.copy()

    new_df = new_df.drop(columns = ['Clip'])
    
    # For loop drops 'likelihood' columns
    for i in range(len(df.columns)):
        if "likelihood" in df.columns[i]:
            new_df = new_df.drop(columns = [df.columns[i]])

    return new_df