import cv2
import opencv_jupyter_ui as jcv2
from tqdm import tqdm

## Loads zebrafish videos into numpy array. 
def load_video_clips(fish_vid_paths):
    """
    Loads the video clips that make up the recording into a single list of arrays. Each array 
    contains the image information for each frame of the recording.

    Parameters:
        fish_vid_paths (list[string,...]): A list of the paths for each of the videos that make up
        the recording
    Returns:
        frame_list (list[ndarray,...]): A list of arrays. Each array contains the image data for each frame
    """
    frame_list = []

    # Iterate through each video clip
    for j in range(len(fish_vid_paths)):
        fish_vid = cv2.VideoCapture(fish_vid_paths[j])

        # Load Video
        vid_length = int(fish_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Iterate through each frame in the clip and add it to the list of frames for the entire recording
        for i in tqdm(range(vid_length), desc = 'Loading Video ' + str(j+1) + ' of ' + str(len(fish_vid_paths))):
            ret, frame = fish_vid.read()
            if(not ret):
                break
            frame_list.append(frame)
            
        # When everything done, release
        # the video capture object
        fish_vid.release()
    return frame_list

## Annotates videos
def annotate_video(pdf, df, frame_list, points, p_cutoff = 0.6, circle_color = (255, 255, 255), error_color = (0, 0, 255)):
    '''
    For each frame in frame_list, annotates the frame with the bodypart labels and the number of the frame in the recording,
    Bodypart labels with a low likelihood are labelled in a different color.

    Parameters:
        pdf (pandas.DataFrame): Dataframe from which to get the positions of each bodypart label

        df (pandas.DataFrame): Full, unaltered DLC DataFrame with likelihood data

        frame_list (list(ndarray,...)): List of images which make up the recording

        points (list(string,...)): List of bodypart label names

        p_cutff (float): The likelihood value below which points are considered to be of 'low certainty'

        circle_color (Tuple(int, int, int)): The color of high certainty bodypart labels

        error_color (Tuple(int,int,int)): The color of low certainty bodypart labels

    Returns:
        disp_frame_list (list(ndarray,...)): List of annotated frames
    '''
    # Dataframe with positions of labels to be  displayed
    disp_pdf = pdf.copy()

    frame_shape = frame_list[0].shape

    # Parameters for the text annotation showing the frame number
    text_font = cv2.FONT_HERSHEY_PLAIN
    text_pos = (0,0)
    text_scale = frame_shape[0]//200
    text_color = (255, 255, 255)
    text_thickness = 3

    # Parameters for the circle annotations of the transformed points
    circle_radius = frame_shape[0]//100
    circle_thickness = -1

    # Format data to use for annotation
    disp_pdf = disp_pdf.round()
    disp_pdf = disp_pdf.astype('int64')

    # Array of annotated frames
    disp_frame_list = []

    ## Apply Annotations to each frame
    for i in tqdm(range(len(frame_list)), desc = 'Preparing Video for Display'):
        frame = frame_list[i]
        
        # Add Frame Text Annotation
        disp_frame = cv2.putText(frame.copy(), str(i), text_pos, text_font, text_scale, text_color, text_thickness, bottomLeftOrigin=True)

        # For each label, add circle annotation to frame
        for point in points:
            # Extract position and likelihood data for the label
            likelihood = df.loc[i].at[point + ' likelihood']
            x = disp_pdf.loc[i].at[point + ' x']
            y = disp_pdf.loc[i].at[point + ' y']

            # Set color for high-certainty points
            color = circle_color
            
            # Print low-uncertaint points and set their color to the color indicated by error_color
            if(likelihood < p_cutoff):
                color = error_color
            
            # Add circle annotation to frame
            disp_frame = cv2.circle(disp_frame, (x, y), circle_radius, color, circle_thickness)

        # Flip y-axis so that image will be displayed with correct orientation
        disp_frame = cv2.flip(disp_frame,0)
        # Add annotated frame to disp_frame_list
        disp_frame_list.append(disp_frame)
    
    return disp_frame_list

## Displays videos in window
def display(disp_frame_list, speeds = [10, 40, 100, 500, 1000], paused=True, s=2, reverse=False, f=0):
    """
    Opens a GUI for the visualization of the transformed data on the given recording. The GUI has a several buttons that can be used to change the playback 
    speed and direction, quit and pause the video, and step through the video frame by frame.

    Parameters:
        disp_frame_list (list[ndarray,...]): List of frame images to be displayed
        
        speeds (list[int,...]): Time spent on each frame (in ms) while the GUI video is unpaused
        
        paused (bool): whether or not to start the GUI video paused
        
        s (int): The index for the value in speeds which will be the initial playback speed
        
        reverse (bool): Whether the video should begin by playing forward (False) or in reverse (True)
        
        f (int): The frame index with which to start the video
    """
    # Keys to be displayed in GUI
    jcv2.setKeys(['q','up','down','space', 'left', 'right'])

    # Display until video is interrupted
    while(len(disp_frame_list) > 0):
        disp_frame = disp_frame_list[f]
        jcv2.imshow("Zebrafish", disp_frame)
        key = jcv2.waitKey(speeds[s])

        # exits video if the Q key is pressed
        if key == ord('q'):
            break
        
        # pauses video if Space key is pressed
        if key == 32:
            paused = not paused
        
        # Speeds up video if Up key is pressed
        if key == 2621440:
            s = min(len(speeds)-1, s + 1)
        
        # Slows down video if Down key is pressed
        if key == 2490368:
            s = max(0, s +- 1)
        
        if(not paused):
            # Plays in Reverse if Left Key is Pressed and Video Is Not Paused
            if key == 2424832:
                reverse = True
            # Plays Forward if Right Key is Pressed and Video is not Paused
            if key == 2555904:
                reverse = False
            if reverse:
                f = max(0, f - 1)
            else:
                f = min(f + 1, len(disp_frame_list) - 1)
        
        else:
            # Steps Backward if Left Key is Pressed and Video is Paused
            if key == 2424832:
                f = max(0, f - 1)
            # Steps Forward if Right Key is Pressed and Video is not Paused
            if key == 2555904:
                f = min(f + 1, len(disp_frame_list) - 1)

    # Closes the GUI
    jcv2.destroyAllWindows()
    return
