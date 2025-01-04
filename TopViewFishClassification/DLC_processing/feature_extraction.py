import pandas as pd
import numpy as np

## Functions for Feature Extraction

# Helper Functions
def extract_distance(point1_x, point1_y, point2_x, point2_y):
    """
    Returns the euclidean distance between the two given points
        
        Parameters:
            point1_x (ndarray type or float): x coordinate of the first point

            point1_y (ndarray type or float): y coordinate of the first point
            
            point2_x (ndarray type or float): x coordinate of the second point

            point2_y (ndarray type or float): y coordinate of the second point

        Returns: (ndarray type or float) Euclidean distance between point1 and point2
    """
    return np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)

def extract_angle(dist_a, dist_b, dist_c):
    """
    For each row in the given data structures, returns the angle at point c created by the line segments from point a to 
    point c and from point c to point b. The angle is calculated using to the Law of Cosines given the distances between 
    point a and point b (dist c), point a and point c (dist b), and point b and point c (dist a).
        
        Parameters:
            dist_a (ndarray type): data structure of distances between point b and point c

            dist_b (ndarray type): data structure of distances between point a and point c
            
            dist_c (ndarray type): data structure of distances between point a and point b

        Returns: (ndarray type) data structure containing angles in radians at point c
    """
    # Initialize array
    angles = pd.Series(np.zeros(dist_a.size))
    
    # Make sure that zero values are ignored and are given an angle value of 0 instead of nan
    non_zero = ~np.isclose(dist_a * dist_b, np.zeros(dist_a.size), atol=1e-8)
    dist_a, dist_b, dist_c = dist_a[non_zero], dist_b[non_zero], dist_c[non_zero]

    # Get the cosine value
    cosine = (dist_a**2 + dist_b**2 - dist_c**2) / (2 * dist_a * dist_b)
    
    # Ensure any -1.0 or 1.0 values are converted to -1 and 1 (np.cosine gets weird about float values)
    cosine[cosine < -1] = -1
    cosine[cosine > 1] = 1

    # Get the angle from the cosine value
    angles[non_zero] = np.arccos(cosine)

    return angles

def extract_angle_sign(point1_x, point1_y, point2_x, point2_y, point3_x, point3_y):
    """
    Given the three points, gives the angle formed by the three points at point 2 a positive
    or negative value depending upon the orientation of the three points with respect to each
    other. The angle is considered to be positive if the vector from point 1 to point 2 lies
    above (in the y-direction) the axis created by the vector from point 2 to point 3. The
    sign is -1 otherwise. If the vectors are oriented in the same direction, the angle 
    is 0.
        
        Parameters:
            point1_x (ndarray type): data structure of x positions for point 1 over time (the rostral-most point)

            point1_y (ndarray type): data structure of y positions for point 1 over time (the rostral-most point)

            point2_x (ndarray type): data structure of x positions for point 2 over time (the central point)

            point2_y (ndarray type): data structure of y positions for point 2 over time (the central point)

            point3_x (ndarray type): data structure of x positions for point 3 over time (the caudal-most point)

            point3_y (ndarray type): data structure of y positions for point 3 over time (the caudal-most point)

        Returns: (ndarray type) data structure containing the angle signs given to each arrangment of points
    """

    ## Conditions for when to return a - sign
    
    # For when the slope between point 1 and point 2 is infinity
    if(np.isclose(point1_x, point2_x, atol = 1e-8)):
        if(point1_y < point2_y and point3_x > point2_x):
            return -1
        elif(point1_y > point2_y and point3_x < point2_x):
            return -1 
    # For slopes between point 1 and point 2 < infinity
    else:
        # Find the line between point 1 and point 2 and use it to determine the sign
        m = (point2_y-point1_y) / (point2_x-point1_x)
        b = point1_y - point1_x*m
        if(point1_x > point2_x and point3_y > m*point3_x + b):
            return -1
        elif(point1_x < point2_x and point3_y < m*point3_x + b):
            return -1
        
    ## Return + sign if none of the above conditions are true
    return 1

def get_angle(point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, signed = True):
    """
    For every set of three points, returns the angle between the three points at point 2. If the
    signed variable is true, will also return signed angles, where an angle is considered 
    to be positive if the vector from point 1 to point 2 lies above (in the y-direction) the 
    axis created by the vector from point 2 to point 3. The angle is -1 otherwise. If the vectors
    are oriented in the same direction, the angle is 0. If they are oriented in opposite directions,
    the angle is (+/-) pi.
        
        Parameters:
            point1_x (ndarray type): data structure of x positions for point 1 over time (the rostral-most point)

            point1_y (ndarray type): data structure of y positions for point 1 over time (the rostral-most point)

            point2_x (ndarray type): data structure of x positions for point 2 over time (the central point)

            point2_y (ndarray type): data structure of y positions for point 2 over time (the central point)

            point3_x (ndarray type): data structure of x positions for point 3 over time (the caudal-most point)

            point3_y (ndarray type): data structure of y positions for point 3 over time (the caudal-most point)
            
            signed (bool): whether or not to return a signed angle for each set of points

        Returns: (ndarray type) data structure containing the angles given to each set of points
    """
        
    # Get the distances between each pair of points 
    dist_a = extract_distance(point1_x, point1_y, point2_x, point2_y)
    dist_b = extract_distance(point2_x, point2_y, point3_x, point3_y)
    dist_c = extract_distance(point1_x, point1_y, point3_x, point3_y)

    # Get the angle at point 2 and then remap the angle values from [0,pi] to [pi,0]
    angles =  np.pi - extract_angle(dist_a, dist_b, dist_c)

    # Get the sign of the angle according to the orientation of the points
    if signed:
        holder = pd.DataFrame({point1_x.name: point1_x, 
                               point1_y.name: point1_y, 
                               point2_x.name: point2_x, 
                               point2_y.name: point2_y, 
                               point3_x.name: point3_x, 
                               point3_y.name: point3_y})
        cols = holder.columns
        angles = np.multiply(angles, holder.apply(lambda x: extract_angle_sign(x[cols[0]], x[cols[1]], x[cols[2]], x[cols[3]], x[cols[4]], x[cols[5]]), axis=1))

    return angles

def get_eye_angle(eye1_x, eye1_y, eye2_x, eye2_y, ref1_x, ref1_y, ref2_x, ref2_y):
    """
    Given the eye points spanning the widest diameter of the eye and a reference vector, finds the angle
    between the reference vector and the vector perpendicular to the segment between the two eye points.
        
        Parameters:
            eye1_x (ndarray type): data structure of x positions for the first eye point over time 

            eye1_y (ndarray type): data structure of y positions for the first eye point over time 

            eye2_x (ndarray type): data structure of x positions for the second eye point over time 

            eye2_y (ndarray type): data structure of y positions for the second eye point over time 

            ref1_x (ndarray type): data structure of x positions for the first referece point over time

            ref1_y (ndarray type): data structure of y positions for the first reference point over time

            ref2_x (ndarray type): data structure of x positions for the second referece point over time

            ref2_y (ndarray type): data structure of y positions for the second reference point over time
            
        Returns: (ndarray type) data structure containing the eye angles of a single eye over time
    """
    # Get vector perpendicular to the widest diameter of the eye. This vector should be
    # pointing away from the first reference point
    if(np.mean(eye1_y) > np.mean(ref1_y)):
        eye_vec = [(-1 * (eye2_y - eye1_y)/(eye2_x - eye1_x)), 1]
    else:
        eye_vec = [((eye2_y - eye1_y)/(eye2_x - eye1_x)), -1]
    
    # Get the vector of the reference points pointing in the caudal direction 
    ref_vec = [1, 1 * (ref2_y - ref1_y) / (ref2_x - ref1_x)]
    
    # Get unit vectors
    eye_vec[0] = eye_vec[0] / np.sqrt(eye_vec[0]**2 + eye_vec[1]**2)
    eye_vec[1] = eye_vec[1] / np.sqrt(eye_vec[0]**2 + eye_vec[1]**2)
    ref_vec[0] = ref_vec[0] / np.sqrt(ref_vec[0]**2 + ref_vec[1]**2)
    ref_vec[1] = ref_vec[1] / np.sqrt(ref_vec[0]**2 + ref_vec[1]**2)

    # Dot product of the reference vector and the vector perpendicular 
    # to the widest diameter of the eye
    cosine = ref_vec[0] * eye_vec[0] + ref_vec[1] * eye_vec[1]

    # Ensure any -1.0 or 1.0 values are converted to -1 and 1 (np.cosine gets weird about float values)
    cosine[cosine < -1] = -1
    cosine[cosine > 1] = 1

    # Return the angle between the reference and vector and the vector perpendicular to the widest
    # diameter of the eye
    return np.arccos(cosine)

# Feature Extraction Function for a Top-View Recording
def top_view_extract_features(df):
    """
    Given the position data for each label on a top-view zebrafish recording over time, extracts
    the angles along the tail and the angles between the midline and the vector perpendicular
    to the largest diameter of each eye
        
        Parameters:
            df: dataframe containing position data. Can be a dataframe that has 'likelihood' columns
            
        Returns: (DataFrame) dataframe of containing extracte features for the whole recording
    """
    
    top_view_cols = ['Tail Tip', 'Lower Tail', 'Mid-Tail', 
                     'Upper Tail', 'SB', 'Head', 'Lower Left Eye', 
                     'Upper Left Eye', 'Lower Right Eye', 'Upper Right Eye']
    
    for col in top_view_cols:
        str_err = ("A label expected by top_view extract_features is missing from the inputted dataframe." 
                   " top_view_extract_features expects both x and y coordinates for the labels: Tail Tip," 
                   " Lower Tail, Mid-Tail, Upper Tail, SB, Head, Lower Left Eye, Upper Left Eye," 
                   " Lower Right Eye, and Upper Right Eye.")
        assert (col + ' x') in df.columns, str_err
        assert (col + ' y') in df.columns, str_err

    features_df = pd.DataFrame()

    # Tail Angles. These are Extracted as Features. These angles have signs indicating which direction the tail is bent in
    mid_lower_tip_angle = get_angle(df['Mid-Tail x'], 
                                    df['Mid-Tail y'], 
                                    df['Lower Tail x'], 
                                    df['Lower Tail y'], 
                                    df['Tail Tip x'], 
                                    df['Tail Tip y'])
    upper_mid_lower_angle = get_angle(df['Upper Tail x'], 
                                      df['Upper Tail y'], 
                                      df['Mid-Tail x'], 
                                      df['Mid-Tail y'], 
                                      df['Lower Tail x'], 
                                      df['Lower Tail y'])
    SB_upper_mid_angle = get_angle(df['SB x'], 
                                   df['SB y'], 
                                   df['Upper Tail x'], 
                                   df['Upper Tail y'], 
                                   df['Mid-Tail x'], 
                                   df['Mid-Tail y'])
    head_SB_upper_angle = get_angle(df['Head x'], 
                                    df['Head y'], 
                                    df['SB x'], 
                                    df['SB y'], 
                                    df['Upper Tail x'], 
                                    df['Upper Tail y'])

    # Eye Angles. These are Extracted as Features. They do not have an angle sign since the eyes do not move 360 degrees
    left_eye_angle = get_eye_angle(df['Upper Left Eye x'], 
                                   df['Upper Left Eye y'], 
                                   df['Lower Left Eye x'], 
                                   df['Lower Left Eye y'], 
                                   df['Head x'], 
                                   df['Head y'], 
                                   df['SB x'], 
                                   df['SB y'])
    right_eye_angle = get_eye_angle(df['Upper Right Eye x'], 
                                    df['Upper Right Eye y'], 
                                    df['Lower Right Eye x'], 
                                    df['Lower Right Eye y'], 
                                    df['Head x'], 
                                    df['Head y'], 
                                    df['SB x'], 
                                    df['SB y'])
    
    # Loading Tail Angles to Dataframe
    features_df['Lower Tail Angle'] = mid_lower_tip_angle
    features_df['Mid-Tail Angle'] = upper_mid_lower_angle
    features_df['Upper Tail Angle'] = SB_upper_mid_angle
    features_df['SB Tail Angle'] = head_SB_upper_angle

    # Loading Eye Angles to Dataframe
    features_df['Left Eye Angle'] = left_eye_angle
    features_df['Right Eye Angle'] = right_eye_angle

    return features_df
