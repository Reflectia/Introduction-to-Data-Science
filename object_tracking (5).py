import cv2
import numpy as np
from matplotlib import pyplot as plt


def CamShift(cap, scale_factor):

    # Read the first frame
    ret, frame = cap.read()
    '''
        WARNING! click ENTER
    '''
    # Resize the frame based on the scale_factor
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    # Set the ROI (Region of Interest)
    x, y, w, h = cv2.selectROI(frame)

    # Initialize the tracker
    roi = frame[y:y + h, x:x + w]
    roi_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

    plt.plot(roi_hist, color='b')
    plt.show()

    roi_hist = cv2.normalize(roi_hist, roi_hist, 50, 250, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Convert the frame to HSV color space, hue, saturation, value(brightness)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 256], 1.5)

        '''
        Apply the CamShift algorithm
        '''
        # Apply the CamShift algorithm
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        # Draw the track window on the frame
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', img2)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


# -------------------------- Головні виклики -----------------------------------
if __name__ == '__main__':

    # Read the video
    cap = cv2.VideoCapture('Videos/V_1.mp4')

    # Set the scale factor (you can adjust this value)
    scale_factor = 1

    # cap = cv2.VideoCapture('Videos/V_2.mp4')
    # cap = cv2.VideoCapture('Videos/V_3.mp4')
    # scale_factor = 0.3

    CamShift(cap, scale_factor)
