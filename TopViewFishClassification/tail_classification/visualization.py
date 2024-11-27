import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import pandas as pd

import sys
import os

# adding helpers to the system path
package_directory = os.getcwd().split(os.pardir)[0]
sys.path.insert(0, package_directory + '/TopViewFishClassification/helpers')
from helpers import pd_rolling

## Visualisations of Random Forest Results 

def display_random_forest_performance(dfs, datas, clf, fish_num, title = None, filter_window = 0, slice = None):
    """
    Compares how the Random Forest Classifier performed in comparison to the human-labelled active/rest labels
    over the course of one recording. 

    Parameters:
        dfs (list[pandas.DataFrame,...]): List of Dataframes containing an 'Active' column with the human labels
        for each recording
        
        datas (list[ndarray,...]); A list of arrays, for each recording, with the formatted data for input 
        into the sklearn classifier
        
        clf: Trained sklearn classifier
        
        fish_num (int): The ID number of the recording to be displayed
        
        filter_window (int): Will filter out frames of activity less than floor(filterwindow/2) before
        displaying the plot. Will do nothing if filter_window is less than 0.
        
        slice ([int, int]): The starting and ending frame indices for the slice of the recording to be
        displayed. If slice is None, the entire recording will be displayed
    """

    # Get the data for the specified recording
    df, data = dfs[fish_num-1], datas[fish_num-1]
    # Run classifier
    pred_i = clf.predict(data)
    # Filter out small intervals if specified
    if filter_window > 0:
        pred_i = pd_rolling(pd.Series(data=pred_i), filter_window, np.median, right_leaning=True).to_numpy()
    y_i = df['Active'].values[:]
    
    # Only display a slice of the recording if specified
    if slice is None:
        slice = [0, len(pred_i)]

    # Plot the comparison between the classifier-labelled data and the human-labelled data
    plt.plot(np.arange(slice[0], slice[1]), pred_i[slice[0]:slice[1]], label='Manual Label')
    plt.plot(np.arange(slice[0], slice[1]), y_i[slice[0]:slice[1]], label='Model Prediction')
    plt.xlabel('Recording Frame Number')
    plt.yticks([-1, 0, 1], ['Unlabelled', 'Rest', 'Active'])
    if title is not None:
        plt.title(title)
    plt.legend()
    return

## Visualizations of Kinematic Behaviors

# Helper Functions
def plot_rot_interval(df, interval, ax, colormap = plt.cm.YlOrRd_r, x_to_y = 2, title = None, max_interval = None, info = False):
    '''
    For a subplot, given by ax, plots the skeleton of the zebrafish over the course of the given interval. A skeleton is plotted
    for each frame within the interval.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the pose data from which to plot the zebrafish skeleton for each frame
        
        interval ([int, int, int]): The 1d array specifying the location of the interval. The first index is the number
        of the frame at which the interval starts and the second index is the number of the frame at which the interval
        ends. The third index indicates from which recording the interval originates

        ax (matplotlib.axes): The subplot to be plotted

        colormap (matplotlib.pyplot.cm): The zebrafish skeletons are colored according to their position in time in the interval.
        The colors used to indicate the temporal position of the skeleton within the interval are chosen from the gradient
        given by this variable

        x_to_y (float): sets the length of the y-axis by specifying the ratio between the maximum width of the zebrafish skeleton
        over the interval and the length of the y-axis. That is, the length of the y-axis will be 2 times x_to_y times the maximum 
        width of the zebrafish skeleton over the interval

        title (string): The title of the subplot

        max_interval (int): Will evenly divide the color gradient specified by colormap into max_interval colors, where each color 
        represents the temporal position of a skeleton within the interval. This parameter is used
        to set a single color gradient for comparing the time course of multiple intervals of size up to max_interval. 
        If max_interval is None, the same number of colors as the number of frames in the inputted interval will be extracted

        info: If true, displays additional info about the interval in the plot. This includes the starting and ending frame numbers of the
        interval and the number of the recording from which the interval originates.
    '''
    ax.axis('off')
    ax.set_aspect('equal')

    # Set subplot title if specified
    if title is not None:
        ax.set_title(title, fontsize=20)
    
    # Extract the pose data for the given interval
    pos = df.values[interval[0]:interval[1], :].transpose()
    pos_x = pos[::2, :]
    pos_y = pos[1::2, :]
    interval_len = pos.shape[1]
    
    # Display the interval info if specified
    if info:
        text = 'interval = ' + str(interval[0]) + ':' + str(interval[1]) + '\n' + 'Fish ' + str(interval[2]+1)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')

    # Extract the colors to be used to indicate the temporal position of each plot
    # from the colormap
    if max_interval is None:
        color_idx = np.linspace(0, 1, num=interval_len)
    else:
        color_idx = np.linspace(0, 1, num=max_interval)

    # Orient the zebrafish skeletons to be roughly parallel to the x-axis.
    # This is done by setting the average position of the head label as the origin, 
    # and then orienting the x-axis parallel to the segment between the average 
    # position of the head and swim bladder labels.
    origin = np.array([np.mean(pos_x[0,:]), np.mean(pos_y[0,:])])
    ref_vec = (np.array([np.mean(pos_x[1,:]), np.mean(pos_y[1,:])]) - origin) 
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    A = np.array([[ref_vec[0], ref_vec[1]],
                 [ref_vec[1], -ref_vec[0]]])
    A = np.linalg.inv(A)

    # Set the length of the y-axis
    y_lim = np.max(np.abs(pos_x - origin[0]))/x_to_y
    if ax is None:
        plt.ylim(-y_lim, y_lim)
    else:
        ax.set_ylim(-y_lim, y_lim)
    
    # Plot the zebfrafish skeleton for each frame over the course of the interval
    for i in reversed(range(interval_len)):
        color = colormap(color_idx[i])
        pose = np.vstack([pos_x[:,i], pos_y[:,i]])
        pose = np.matmul(A, pose - np.array([origin]).transpose())
        ax.plot(pose[0,:], pose[1,:], 'o-', color = color)
    return

def get_rot_interval_data(df, interval, ax, x_to_y, color = 'black', max_interval=None, int_padding=3):
    """
    For a subplot, given by ax, generates an animation of the zebrafish skeleton over the course of the interval
    
    Parameters:
        df (pandas.DataFrame): The dataframe containing the pose data from which to plot the zebrafish skeleton for each frame
        
        interval ([int, int, int]): The 1d array specifying the location of the interval. The first index is the number
        of the frame at which the interval starts and the second index is the number of the frame at which the interval
        ends. The third index indicates from which recording the interval originates

        ax (matplotlib.axes): The subplot to be animated

        color: The color of the zebrafish skeletons to be plotted

        x_to_y (float): sets the length of the y-axis by specifying the ratio between the maximum width of the zebrafish skeleton
        over the interval to the length of the y-axis. That is, the length of the y-axis will be 2 times x_to_y times the maximum 
        width of the zebrafish skeleton over the interval

        max_interval (int): Allows the interval to be compared to other intervals of up to max_interval in the same animation by
        padding the given interval up to max_interval with the interval's final frame

        int_padding (int): Adds int_padding frames before and after the interval in order to make the behavior displayed in the
        interval easier to visualize

    Returns:
        x_data (ndarray): The x positions of the bodypart labels that make up the skeleton to be animated
        y_data (ndarray): the y positions of the bodypart labels that make up the skeleton to be animated
        plot (matplotlib.Lines.Line2D): The lines representing the zebrafish skeleton to be animated 
    """

     # Extract the pose data for the given interval
    interval_min = max(0, interval[0]-int_padding)
    interval_max = min(interval[1]+int_padding, df.values.shape[0])
    pos = df.values[interval_min:interval_max, :].transpose()
    pos_x = pos[::2, :]
    pos_y = pos[1::2, :]

    # Specify what the length of the interval should be
    interval_len = pos.shape[1]
    if max_interval is None:
        max_interval = interval_len

    # Orient the zebrafish skeletons to be roughly parallel to the x-axis.
    # This is done by setting the average position of the head label as the origin, 
    # and then orienting the x-axis parallel to the segment between the average 
    # position of the head and swim bladder labels.
    origin = np.array([np.mean(pos_x[0,:]), np.mean(pos_y[0,:])])
    ref_vec = (np.array([np.mean(pos_x[1,:]), np.mean(pos_y[1,:])]) - origin) 
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    A = np.array([[ref_vec[0], ref_vec[1]],
                 [ref_vec[1], -ref_vec[0]]])
    A = np.linalg.inv(A)

    # Set the length of the y-axis
    x_lim = np.max(np.abs(pos_x - origin[0]))
    y_lim = x_lim/x_to_y
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlim(-x_lim * 0.1, x_lim *1.1)
    
    x_data = []
    y_data = []

    # Generate and save the skeleton plots for animation of the subplot into arrays
    for i in range(0, interval_len):
        pose = np.vstack([pos_x[:,i], pos_y[:,i]])
        pose = np.matmul(A, pose - np.array([origin]).transpose())
        if i == 0:
            plot = ax.plot(pose[0,:], pose[1,:], 'o-', color = color)[0]
        x_data.append(pose[0,:])
        y_data.append(pose[1,:])
    # Add padding up to max_interval
    for i in range(interval_len, max_interval):
        x_data.append(pose[0,:])
        y_data.append(pose[1,:])

    return x_data, y_data, plot

def plot_feature_trajectories(df, interval, ax=None, title = None, colormap = plt.cm.Set2):
    """
    Plots the trajectories for the extracted features of an interval one on top of the other. Is used as a helper function but
    can also be used on its own. (Note this expects the extrated features to be angles. Will have to be modified if other
    features are added)

    Parameters:
        df (pandas.DataFrame): The dataframe containing the pose data from which to plot the zebrafish skeleton for each frame
        
        interval ([int, int, int]): The 1d array specifying the location of the interval. The first index is the number
        of the frame at which the interval starts and the second index is the number of the frame at which the interval
        ends. The third index indicates from which recording the interval originates

        ax (matplotlib.axes): If used as a helper function to generate a larger plot, the subplot on which to plot
        the feature trajectories

        title (string): Title of the plot

        colormap (matplotlib.pyplot.cm): The colormap that specifies the colors for each trajectory
    """

    # If not acting as a helper function, generate a figure to hold the plot
    if ax is None:
        fig, ax = plt.subplots()

    # Add title if specified
    if title is not None:
        ax.set_title(title)

    # Format the plot
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Frame Number')
    
    # Arrange the angles one on top of the other in a single plot
    angles = df.values[interval[0]:interval[1], :-1].transpose()
    shift = 2 * np.pi
    angles = np.array([angles[i,:] + shift*i for i in range(angles.shape[0])])
    
    # Iterate through and plot each feature trajetory
    for i in range(angles.shape[0]):
        ax.plot(angles[i], color=colormap(i), label=df.columns[i])
        ax.legend(loc = 'upper left')

    return

# Visualisation Functions
def plot_animated_intervals(dfs, intervals, max_interval, save_as, ncols = 1, color = 'black', x_to_y = 2, fig_width = 10, subtitles = None, title = None, int_padding = 3):
    """
    Generates and saves an animation comparing the progressions of the zebrafish skeleton between several different intervals. The animation
    is saved as both a gif and a .mov file.

    Parameters:
        dfs (list[pandas.DataFrame,...]): A list of the dataframes containing the pose data from which to plot the zebrafish 
        skeleton for each frame
        
        intervals (ndarra): Array containing the information for the intervals to be displayed

        save_as (string): path to where the gif animation should be saved

        ncols: The number of columns of subplots

        color (string or Tuple(int, int, int)): The color of the zebrafish skeletons to b e plotted

        x_to_y (float): sets the length of the y-axis by specifying the ratio between the maximum width of the zebrafish skeleton
        over the interval to the length of the y-axis. That is, the length of the y-axis will be 2 times x_to_y times the maximum 
        width of the zebrafish skeleton over the interval

        fig_width (float): The width of the entire plot to be animated

        subtitles (list[string,...]): List of containing the titles for each subplot. Must be equal the number of intervals to be plotted

        title (string): The title for the entire plot

        max_interval (int): Allows all intervals to be compared to other intervals of up to max_interval in the same animation by
        padding all intergals up to max_interval with the intervals' final frame.

        int_padding (int): Adds int_padding frames before and after each interval in order to make the behavior displayed in the
        interval easier to visualise
    """

    # Sets the number of columns in the subplot grid
    shape = (math.ceil(len(intervals)/ncols), ncols)

    # Set the format of the plot
    fig_height = shape[0] * fig_width/(shape[1] * x_to_y) * 2
    fig, axs = plt.subplots(shape[0], shape[1], squeeze=False, figsize = (fig_width, fig_height))
    for i in range(0, shape[0] * shape[1]):
        ax = axs[i//shape[1], i%shape[1]]
        ax.axis('off')
        ax.set_aspect('equal')
        if subtitles is not None and i < len(intervals):
            ax.set_title(subtitles[i])

    x_datas = []
    y_datas = []
    plots = []
    max_interval = int(max_interval + int_padding*2)

    # Put together the animation
    for i in range(len(intervals)):
        interval = intervals[i]
        plot_ind = [i//shape[1], i%shape[1]]
        df = dfs[interval[2]]
        interval = interval[:2]
        x_data, y_data, plot = get_rot_interval_data(df, interval, axs[plot_ind[0], plot_ind[1]], x_to_y, color=color, max_interval=max_interval, int_padding=int_padding)
        x_datas.append(x_data)
        y_datas.append(y_data)
        plots.append(plot)

    # A helper functino for setting up the animation. This
    # has to be passed to FuncAnimation. See
    # https://matplotlib.org/stable/users/explain/animations/animations.html
    # for more information
    def update(frame):
        for i in range(0, len(intervals)):
            plot = plots[i]
            x_data = x_datas[i][frame]
            y_data = y_datas[i][frame]
            plot.set_xdata(x_data)
            plot.set_ydata(y_data)
        return tuple(plots)

    # Set  overall figure title if specified  
    if title is not None:
        fig.suptitle(title, size = 'xx-large')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout()
    
    # Generate Animation
    anim = animation.FuncAnimation(fig=fig, func=update, frames=max_interval, interval=200)
    
    # Save animation as gif and mov
    anim.save(save_as + '.gif', writer = animation.ImageMagickWriter())
    anim.save(save_as + '.mov')
    print('Your animation was successfully saved in ' + save_as + '.gif and ' + save_as + '.mov!')
    plt.close()
    return

def plot_intervals(dfs, intervals, ncols = None, colormap = plt.cm.YlOrRd_r, x_to_y = 2, fig_width = 10, subtitles = None, title = None, colorbar=False, max_interval = None, info=False):
    '''
    Plots the progression of the zebrafish skeletons over several intervals, with each interval constituting a different subplot

    Parameters:
        dfs (list[pandas.DataFrame,...]): A list of the dataframes containing the pose data from which to plot the zebrafish 
        skeleton for each frame
        
        intervals (ndarra): Array containing the information for the intervals to be displayed

        ncols: the number of columns of subplots

        colormap (matplotlib.pyplot.cm): The zebrafish skeletons are colored according to their position in time in the interval.
        The color used to indicate the temporal position of the skeleton within the interval is chosen from the gradient
        given by this variable

        x_to_y (float): sets the length of the y-axis by specifying the ratio between the maximum width of the zebrafish skeleton
        over the interval to the length of the y-axis. That is, the length of the y-axis will be 2 times x_to_y times the maximum 
        width of the zebrafish skeleton over the interval

        subtitles (list[string,...]): List of containing the titles for each subplot. Must be equal the number of intervals to be plotted

        title (string): The title of the entire figure

        colorbar (bool): If True, adds a colorbar to the plot

        max_interval (int): Will evenly divide the color gradient specified by colormap into max_interval colors, where each color 
        represents the temporal position of a skeleton within an interval. This parameter is used
        to set a single color gradient for comparing the time course of multiple intervals of size up to max_interval. 
        If max_interval is None, the color gradient will be divided evenly, extracting the same number of colors as the
        number of frames in the inputted interval

        info: If true, displays additional info about each interval in the plot. This includes the starting and ending frame numbers of the
        interval and the number of the recording from which the interval originates.
    '''
    # Set Figure Format
    shape = (math.ceil(len(intervals)/ncols), ncols)
    fig_height = shape[0] * fig_width/(shape[1] * x_to_y) * 2
    fig, axs = plt.subplots(shape[0], shape[1], squeeze=False, figsize = (fig_width, fig_height))
    
    # Set title if specified
    if title is not None:
        fig.suptitle(title, size = 'xx-large')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout()

    # Plot the subplot for each interval
    for i in range(len(intervals)):
        if subtitles is None:
            subtitle = None
        else:
            subtitle = subtitles[i]
        interval = intervals[i]
        plot_ind = [i//shape[1], i%shape[1]]
        df = dfs[interval[2]]
        plot_rot_interval(df, interval, colormap = colormap, x_to_y = x_to_y, ax = axs[plot_ind[0], plot_ind[1]], title=subtitle, max_interval=max_interval, info=info)
    
    # If indicated, dd a colorbar to the plot
    if colorbar:
        norm = mcolors.Normalize(vmin=0,vmax=max_interval)
        cmap = plt.get_cmap(colormap, max_interval)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ticks = np.linspace(0, max_interval, 10).astype(int)
        boundaries = np.arange(-0.5, max_interval+1, 1)
        fig.colorbar(sm, ax = axs[-1,:], location='bottom', ticks=ticks, boundaries=boundaries, shrink = 0.3, pad = 0.005, label = 'Frame of Interval')

    # Make any extra subplots in the grid blank
    for i in range(len(intervals), shape[0] * shape[1]):
        axs[i//shape[1], i%shape[1]].axis('off')
    return

def compare_intervals(dfs, pdfs, intervals, title=None, subtitles=None, colormap_a = plt.cm.YlOrRd_r, colormap_b = plt.cm.Set2, info = False, x_to_y=2, fig_size = (15, 10)):
    '''
    For each inputted interval, generate a subplot with a progression of fish skeletons for each frame in the interval and a subplot with the trajectories
    for each extracted feature
    
    Parameters:
        dfs (list[pandas.DataFrame,...]): A list of the dataframes containing the extracted features from which to plot the feature
        trajectories

        pdfs (list[pandas.DataFrame,...]): A list of the dataframes containing the pose data from which to plot the zebrafish 
        skeleton for each frame

        intervals (ndarra): Array containing the information for the intervals to be displayed

        title (string): The title of the entire figure

        subtitles (list[string,...]): List containing the titles for each pair of subplots. Must be equal the number of intervals to be plotted

        colormap_a (matplotlib.pyplot.cm): The zebrafish skeletons are colored according to their position in time in the interval.
        The colors used to indicate the temporal position of the skeleton within the interval are chosen from the gradient
        given by this variable

        colormap (matplotlib.pyplot.cm): The colormap that specifies the colors for each trajectory

        info: If true, displays additional info about each interval in the plot. This includes the starting and ending frame numbers of the
        interval and the number of the recording from which the interval originates.

        x_to_y (float): For the skeleton subplots, sets the length of the y-axis by specifying the ratio between the maximum width 
        of the zebrafish skeleton over the interval to the length of the y-axis. That is, the length of the y-axis will be 2 times 
        x_to_y times the maximum width of the zebrafish skeleton over the interval

        figsize (Tuple(int, int)): The size of the entire figure
    '''
    interval_num = intervals.shape[0]
    fig, axs = plt.subplots(interval_num, 2, squeeze=False, figsize=fig_size)

    # Plot the skeleton progression for the interval
    for i in range(interval_num):
        ax = axs[i, 0]
        interval = intervals[i]
        pdf = pdfs[interval[2]]
        # Add a subplot title to this plot if indicated
        if subtitles is None:
            subtitle = None
        else:
            subtitle = subtitles[i]
        plot_rot_interval(pdf, interval, ax=ax, colormap=colormap_a, info=info, x_to_y=x_to_y, title=subtitle)

    # Plot the angle trajectories for the interval
    for i in range(interval_num):
        ax = axs[i, 1]
        interval = intervals[i]
        df = dfs[interval[2]]
        plot_feature_trajectories(df, interval, ax=ax, colormap=colormap_b)

    # Set the title if indicated and set the format
    if title is not None:
        fig.suptitle(title, size = 'xx-large')
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        fig.tight_layout()
    return

def plot_clusters(intervals, medoids, clusters, correlation, state_titles):
    """
    Plots three graphs showing a low-dimensional representation of how the inputted intervals are arranged relative 
    to each other according to the inputted distance matrix, "correlation". The graphs are then colored in different 
    ways to illustrate how different properties of the intervals relate ot each other. The first graph is colored according
    to which cluster each interval belongs to. The second graph is colored according to which recording
    the interval originates from. The third graph is colored according to the length of each interval.
    
    Parameters:
        intervals (ndarray): a list with information for all the intervals to be plotted    
    
        medoids (list[int,...]): A list of the row indices of the behavioral state medoids in intervals
        
        clusters (list[list[int,...],...): A list of lists. Each list contains the indices, within
        the parameters intervals, for the intervals of each cluster
        
        correlation (ndarray): A matrix of the dynamic time warping distance between each pair of 
        intervals. The rows and columns of this matrix correspond to interval specified by the 
        row with the same index in the array, intervals
        
        state_titles (list[string,...]): A list of the behavioral state names. Should be the same length as medoids
    """

    # Set the format and title for the whole figure
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize = (15, 5))
    fig.suptitle('UMAP Low-Dimensional Embedding of DTW Distances', y = 1.1, size = 'xx-large')
    
    # Plot the first subplot which shows how the data is clustered
    ax = axs[0]
    ax.axis('off')
    ax.title.set_text('According to Behavioral State')

    for i in range(len(clusters)):
        cluster = correlation[clusters[i]][:]
        color = plt.cm.Paired(2*i)
        ax.scatter(cluster[:,0], cluster[:,1], color=color, label = state_titles[i])

    # Mark the medoids for each sluster as stars
    for i in range(len(medoids)):
        medoid = correlation[medoids[i]][:]
        medoid_color = plt.cm.Paired(2*i+1)
        ax.scatter(medoid[0], medoid[1], marker = '*', s = 200, color=medoid_color, edgecolors='black')
    ax.legend()

    # Plot the second subplot which shows from which recording each interval originates
    ax = axs[1]
    ax.axis('off')
    ax.title.set_text('According to Fish')
    for i in range(max(intervals[:,2])+1):
        fish = correlation[intervals[:, 2] == i][:]
        color = plt.cm.Dark2(i)
        ax.scatter(fish[:,0], fish[:,1], color=color, label = 'Fish ' + str(i+1))
    ax.legend()

    # Plot the third subplot which shows how long each interval is
    ax = axs[2]
    ax.axis('off')
    ax.title.set_text('According to Interval Length')
    fps = 25
    color = (intervals[:,1] - intervals[:,0]) / fps
    map = ax.scatter(correlation[:,0], correlation[:,1], c=color, cmap = plt.cm.nipy_spectral, norm=mcolors.LogNorm())
    cbar = fig.colorbar(map, ax=axs[2], label='Interval Length (s)', ticks = np.arange(0, 4, 0.5))
    cbar.ax.set_yticklabels(np.arange(0,4,0.5))

    return