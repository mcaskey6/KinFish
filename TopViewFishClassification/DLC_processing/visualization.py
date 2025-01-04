import matplotlib.pyplot as plt
import math

## Functions for Data Visualization

def plot_trajectories(df_arr, ncols = 2, figsize=(20,30), df_names = None, y_axis = None, legend = False):
    """
    Displays the trajectories of each bodypart label over time. This can be applied to more 
    than one DataFrame at a time for comparison of trajectories. 
        
        Parameters:
            df_arr (list [pandas.DataFrame,...]): A list of the DataFrames to be displayed. If only wish to display 
            one DataFrame pass a list of size one.
            
            ncols (int): Optional parameter. Sets the number of columns in the grid of subplots. Default is 2.
            
            figsize (tuple [int, int]): Optional parameter. Sets the size of the entire plot. Default is (20, 30)
            
            df_names (list [string,...]): Optional parameter. List of names representing each DataFrame in df_arr.
            Will be used in figure title. If no names are specified, each figure will be named 'DataFrame i' where
            i is some integer. The title will be of the form 'DataFrame 0 vs. DataFrame 1 vs. etc'.

            y_axes (string): y axis label. If None, will be set to 'Position'
            
            legend (bool): Optional Parameters. If legend is True, a legend will be used to name the trajectories 
            of each subplot instead of a subplot title. If False, each subplot will have a title of the form
            'ColName 0 vs. ColName 1 vs. etc' where each column name is taken from the column names of the 
            corresponding DataFrame.
    """
    arrs = []
    lengths = []
    suptitle = ''

    for i in range(len(df_arr)):
        
        # Generates the title of the entire figure
        if df_names is None:
            name = 'DataFrame '+ str(i)
        else:
            name = df_names[i]
        if i > 0:
            suptitle = suptitle + ' vs. ' + name
        else:
            suptitle = name
        
        # Converts each dataframe to numpy array for matplotlib
        arrs.append(df_arr[i].copy().to_numpy())
        
        # Saves number of columns in each dataframae
        lengths.append(arrs[i].shape[1])
        
    # Sets format for entire figure
    fig, axs = plt.subplots(nrows=math.ceil(max(lengths)/ncols), ncols=ncols, figsize=figsize, squeeze=False)

    axs_titles = [''] * max(lengths)
    
    # Generates each subplot
    for l, (df, arr) in enumerate(zip(df_arr, arrs)):

        # Plot the trajectories for each column in the selected dataframe.
        for i in range(lengths[l]):
            
            # Subplots are filled from left to right and then up to down
            # for each dataframe selected
            ax = axs[i//ncols, i%ncols]
            title = df.columns[i]
            ax.plot(arr[:,i], label = title)

            # Generates the legend/subplot title for each subplot, including
            # subplots containing more than one trajectory
            if(not legend):
                if(axs_titles[i] == ''):
                    axs_titles[i] = title
                else:
                    axs_titles[i] = axs_titles[i] + ' vs. ' + title
                ax.set_title(axs_titles[i])
            else:
                ax.legend(loc = 'upper right')
            ax.set_xlabel("Frame")
            
            # Set y axis label
            if(y_axis is None):
                ax.set_ylabel("Position")
            else:
                ax.set_ylabel(y_axis)
    
    ## Make subplots that aren't used blank
    for i in range(max(lengths), axs.shape[0]*axs.shape[1]):
        axs[i//ncols, i%ncols].set_axis_off()
    
    # More formatting of entire figure. Specifically setting the spacing
    # between each subplot and setting the position of the overall figure title
    fig.tight_layout()
    fig.suptitle(suptitle, y = 1.01, size = 'xx-large')

    plt.show()

    return