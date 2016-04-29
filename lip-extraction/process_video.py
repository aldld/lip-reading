import numpy as np
from scipy.io import savemat
import sys, os

import cv2

DEBUG = True

# Mouth detection cascade classifier.
cc_mouth = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
cc_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def rect_area(rect):
    """ Returns the area of the rectangle rect. """
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

def uniform_rect(mouth, face, width, height):
    """ Returns a rectangle with the given width and height, centred as closely as possible to the centre of the mouth,
        shifted upwards so as to lie completely within the face. """
    if mouth is None:
        return None

    mc_x, mc_y = mouth[0] + 0.5 * mouth[2], mouth[1] + 0.5 * mouth[3] # Mouth center point.

    rect_bottom = mc_y + 0.5 * height

    rect_x = mc_x - 0.5 * width
    rect_y = mc_y - 0.5 * height - max(0, rect_bottom - (face[2] + face[3]))

    return [int(round(i)) for i in [rect_x, rect_y, width, height]]

def locate_face(image, minNeighbors=5, scaleFactor=1.05):
    """ Returns the largest (by area) rectangle corresponding to a detected face. """
    rects = cc_face.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return max(rects, key=rect_area)

def locate_mouth(image, minNeighbors=10, scaleFactor=1.05):
    """ Returns a list of candidate rectangles found by the mouth detector. """
    rects = cc_mouth.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return rects

def highlight_rect(image, rect, color=(125, 125, 25), thickness=1):
    """ Highlights the given rectangle in the given image. """
    return cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness)

def process(in_path, out_path, mouth_height=50, mouth_width=50, frame_dur=1, capture_frame=None, out_img=None, show_boxes=True):
    """ Processes the video file given by in_path, outputting a .mat file at out_path containing the regions of the
        frames of the original video featuring the speaker's mouth.

        TODO: Refactor this method.
    """
    # Get video capture from in_path.
    vc = cv2.VideoCapture(in_path)    

    rval, frame = vc.read() if vc.isOpened() else (False, None)

    mouth_images = []

    #import pdb; pdb.set_trace()

    if rval:
        mouths = np.empty((0, mouth_height, mouth_width, frame.shape[2]))
    else:
        return # Skip this video since CV2 can't open it

    frame_no = 0
    while rval:
        if DEBUG:
            # Copy of original frame, for annotating.
            image = frame.copy()

        try:
            face_rect = locate_face(frame)
        except ValueError:
            print "No face found for %s at frame %d. Skipping." % (in_path, frame_no)
            vc.release()
            return # Skip this video.

        if DEBUG:
            highlight_rect(image, face_rect, color=(255,255,255), thickness=2)

        mouth_rects = locate_mouth(frame)
        mouth = uniform_rect(select_mouth_candidate(mouth_rects, face_rect), face_rect, 50, 50)
        if not mouth:
            print "No face found for %s at frame %d. Skipping." % (in_path, frame_no)
            vc.release()
            return # Skip this video.

        mouth_image = frame[mouth[1]:(mouth[1] + mouth[3]), mouth[0]:(mouth[0] + mouth[2]), :]
        mouth_images.append(mouth_image)        

        if DEBUG:
            highlight_rect(image, mouth, color=(0,0,0), thickness=2) 
            #cv2.imshow('Frame', mouth_image)
            cv2.imshow('Frame', image if show_boxes else frame)

            if frame_no == capture_frame:
                cv2.imwrite(out_img, image if show_boxes else frame)
                return

            cv2.waitKey(frame_dur)

        rval, frame = vc.read()
        frame_no += 1

    vc.release()

    mouths = np.asarray(mouth_images)

    savemat(out_path, {"mouths": mouths})


def process_all(data_dir, include=set(), max_videos=np.inf, verbose=False):
    num_processed = 0

    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)

        if not speaker_dir in include or not os.path.isdir(speaker_path):
            continue

        print 'current speaker: %s' % speaker_dir

        for f_name in os.listdir(speaker_path):
            if f_name.endswith('.mpg'):
                if num_processed >= max_videos:
                    return

                name = f_name.split('.')[0]
                in_path = os.path.join(speaker_path, f_name)                
                out_path = os.path.join(speaker_path, name+'.mat')
                if verbose:
                    print f_name
                process(in_path, out_path)        
                num_processed += 1

    print "\nFinished processing %d videos." % num_processed

if __name__ == '__main__':
    # Generate images for paper.
    if len(sys.argv) == 2 and sys.argv[1] == "gen_imgs":
        vid1 = "/Users/eric/Programming/prog_crs/lip-reading/data/grid/s1/video/pgak5a.mpg"
        vid2 = "/Users/eric/Programming/prog_crs/lip-reading/data/grid/s4/video/bbbf3n.mpg"

        out_file = "/Users/eric/temp/vid"

        process(vid1, out_file, frame_dur=1, capture_frame=7, out_img="s1.jpg", show_boxes=False)
        process(vid2, out_file, frame_dur=1, capture_frame=11, out_img="s4.jpg", show_boxes=False)

        process(vid1, out_file, frame_dur=1, capture_frame=7, out_img="s1_boxes.jpg", show_boxes=True)
        process(vid2, out_file, frame_dur=1, capture_frame=11, out_img="s4_boxes.jpg", show_boxes=True)

        exit()

    #include = {'s2'}
    #if len(sys.argv) == 2:
    video_path = '/ais/gobi4/freud/temp/data'
    s, f = map(int, sys.argv[1].split(','))
    include = {'s%s'%i for i in xrange(s,f+1)}
    print 'processing ', include
    process_all(video_path, include)
    
    #include = {'s17', 's18'}
    #process_all(sys.argv[1], include, max_videos=(np.inf if len(sys.argv) < 3 else int(sys.argv[2])))
