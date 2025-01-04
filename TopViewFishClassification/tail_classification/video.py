from tqdm import tqdm
import cv2
import imageio

## Functions for Creating and Saving Recordings of Intervals
def load_video_clips(fish_vid_paths):
    """
    Loads the video clips that make up the recording into a single list of arrays. Each array 
    contains the image information for each frame

    Parameters:
        fish_vid_paths (list[string,...]): A list of the paths for each of the videos that make up
        the recording
    Returns:
        frame_list (list[ndarray,...]): A list of arrays. Each array contains the image data for each frame
    """

    frame_list = []
    for j, path in enumerate(fish_vid_paths):
        fish_vid = cv2.VideoCapture(path)

        # Load Video
        vid_length = int(fish_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in tqdm(range(vid_length), desc = 'Loading Video ' + str(j+1) + ' of ' + str(len(fish_vid_paths))):
            ret, frame = fish_vid.read()
            if(not ret):
                break
            frame_list.append(frame)
            
        # When everything done, release
        # the video capture object
        fish_vid.release()
    return frame_list

def annotate_recording(pdf, frame_list, circle_color = (255, 255, 255)):
    '''
    For each frame in frame_list, annotates the image with the bodypart labels and the number of the frame in the recording.

    Parameters:
        pdf (pandas.DataFrame): Dataframe from which to get the positions of each bodypart label

        frame_list (list(ndarray,...)): List of images which make up the recording

        circle_color (Tuple(int, int, int)): The color of high certainty bodypart labels

    Returns:
        disp_frame_list (list(ndarray,...)): List of annotated frames
    '''

    # Position data dataframe to be displayed
    disp_pdf = pdf.copy()

    frame_shape = frame_list[0].shape

    # Parameters for the text annotation signifying the frame number
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add Frame Text Annotation
        disp_frame = cv2.putText(frame.copy(), str(i), text_pos, text_font, text_scale, text_color, text_thickness, bottomLeftOrigin=True)

        # For each label, add circle annotation to frame
        for j in range(len(pdf.columns)//2):
            x = disp_pdf.iloc[i, j*2]
            y = disp_pdf.iloc[i, j*2+1]
            
            # Add circle annotation to frame
            disp_frame = cv2.circle(disp_frame, (x, y), circle_radius, circle_color, circle_thickness)

        # Flip y-axis so that image will be displayed with correct orientation
        disp_frame = cv2.flip(disp_frame,0)
        # Add annotated frame to disp_frame_list
        disp_frame_list.append(disp_frame)
    return disp_frame_list

def save_interval_as_video(disp_frame_list, vid_name):
    """
    Saves a clip of a specified interval of a recordingg as a .move file
    Parameters:
        disp_frame_list (list(ndarray,...)): List of annotated frames that
        make up the interval
        vid_name (string): the path where the .mov file will be stored
    """
    frame_shape = disp_frame_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(vid_name + '.mov', fourcc, 15.0, frame_shape[::-1], True)
    for frame in disp_frame_list:
        out.write(frame)
    return

def save_interval_as_gif(disp_frame_list, gif_name):
    """
    Saves a clip of a specified interval of a recordingg as a .move file
    Parameters:
        disp_frame_list (list(ndarray,...)): List of annotated frames
        that make up the interval
        gif_name (string): the path where the .mov file will be stored
    """
    imageio.mimsave(gif_name + '.gif', disp_frame_list,  loop=0, fps=5)
    return