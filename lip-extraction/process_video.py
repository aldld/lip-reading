import numpy as np
import cv2

# Path to GRID corpus video data.
DATA_DIR = "/Users/eric/Programming/prog_crs/lip-reading/data/grid/video"

#TEST_FN = "/Users/eric/Programming/prog_crs/lip-reading/data/grid/video/s1/bwbg8n.mpg"
TEST_FN = "/Users/eric/Programming/prog_crs/lip-reading/data/grid/video/s1/pgid4n.mpg"

# Mouth detection cascade classifier.
cc_mouth = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
cc_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def rect_area(rect):
    """ Returns the area of the rectangle rect. """
    #return (rect[2] - rect[0]) * (rect[3] - rect[1])
    return rect[2] * rect[3]

def rect_overlap(rect1, rect2):
    """ Returns the area of the intersection of rect1 and rect2. """
    a_x1, a_y1 = rect1[0], rect1[1]
    a_x2, a_y2 = (rect1[0] + rect1[2], rect1[1] + rect1[3])

    b_x1, b_y1 = rect2[0], rect2[1]
    b_x2, b_y2 = (rect2[0] + rect2[2], rect2[1] + rect2[3])

    x_overlap = max(0, min(a_x2, b_x2) - max(a_x1, b_x1))
    y_overlap = max(0, min(a_y2, b_y2) - max(a_y1, b_y1))
    return x_overlap * y_overlap

def rect_delta(rect1, rect2):
    """ Returns the percentage of rect1 that lies outside of rect2. """
    rect1_area = rect_area(rect1)

    return float(rect1_area - rect_overlap(rect1, rect2)) / rect1_area

def select_mouth_candidate(candidates, face, delta=0.3):
    """ Chooses the mouth candidate whose y-midpoint is the greatest (i.e. lowest in the image) such that no more than
        delta*(100 percent) of the candidate lies outside of the face rectangle. """
    best_candidate = None
    best_midpoint = -np.inf # Vertical midpoint of best candidate.
    for candidate in candidates:
        candidate_midpoint = candidate[1] + 0.5*candidate[3]
        if candidate_midpoint > best_midpoint and rect_delta(candidate, face) < delta:
            best_candidate = candidate
            best_midpoint = candidate_midpoint

    return best_candidate

def locate_face(image, minNeighbors=5, scaleFactor=1.05):
    """ Returns the largest (by area) rectangle corresponding to a detected face. """
    rects = cc_face.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return max(rects, key=rect_area)

def locate_mouth(image, minNeighbors=10, scaleFactor=1.05):
    """ Returns a list of candidate rectangles found by the mouth detector. """
    rects = cc_mouth.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    #return max(rects, key=rect_area)
    return rects

def highlight_rect(image, rect, color=(125, 125, 25), thickness=1):
    """ Highlights the given rectangle in the given image. """
    return cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness)

if __name__ == "__main__":
    # Get video capture.
    vc = cv2.VideoCapture(TEST_FN)
    #vc = cv2.VideoCapture(0)

    rval, frame = vc.read() if vc.isOpened() else (False, None)

    while rval:
        image = frame

        face_rect = locate_face(image)
        highlight_rect(image, face_rect, color=(255,255,255), thickness=2)

        rects = locate_mouth(image)
        mouth = select_mouth_candidate(rects, face_rect)
        highlight_rect(image, mouth, color=(0,0,0), thickness=2)

        cv2.imshow('Frame', image)
        cv2.waitKey(1)

        rval, frame = vc.read()

    vc.release()

