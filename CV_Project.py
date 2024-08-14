# Imports
import cv2
import numpy as np
from sklearn.metrics import pairwise

# Global variables
background = None
accumulated_weight = 0.5

roi_top = 80
roi_bottom = 300
roi_right = 300
roi_left = 600

# Functions
def calc_accum_avg(frame,accumulated_weight):
    
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)
    
    
def segments(frame,threshold_min=25):
    
    diff = cv2.absdiff(background.astype('uint8'),frame)
    
    ret,thresholded = cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    
    image,contours,hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    else:
        # Assuming the largest external contour in roi is the hand
        hand_segment = max(contours,key=cv2.contourArea)
    
        return (thresholded,hand_segment) 
    
    
def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    
    # Extract extreme points
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # Calculate center of the hand
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # Reshape the center and extreme points to 2D
    center = np.array([[cX, cY]])
    extreme_points = np.array([left, right, top, bottom])

    # Calculate the Euclidean distances
    distance = pairwise.euclidean_distances(center, extreme_points)[0]

    max_distance = distance.max()
    
    # Define a circular ROI based on the maximum distance
    radius = int(0.9 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Use bitwise AND to apply the mask to the thresholded image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Find contours in the circular ROI
    image,contours,hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    count = 0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        out_of_wrist = (cY + (cY * 0.25)) > (y + h)
        limit_points = (circumference * 0.25) > cnt.shape[0]

        if out_of_wrist and limit_points:
            count += 1
    return count


# Main code
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open the camera.")
    exit()

num_frames = 0

while True:
    
    ret,frame = cam.read()
    
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if num_frames < 60:
        
        calc_accum_avg(gray,accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy, 'WAIT. GETTING BACKGROUND', (200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        
        hand = segments(gray)
        
        if hand is not None:
            
            thresholded, hand_segment = hand
            
            # Draw contours around real hand in live stream
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
    
            fingers = count_fingers(thresholded,hand_segment)
        
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top), (roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    cv2.imshow('Finger Count', frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
    