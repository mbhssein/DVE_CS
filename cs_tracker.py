import os
import skimage
import sys
from matplotlib import pyplot as plt
from time import sleep

import cv2

from dve_utils import SVMDetector, Tracker
from ml import Classifier

# object tracking parameters
MAX_MOVE_DIST = 200
MAX_SKIP_FRAMES = 8
SLIDING_WINDOW_SIZE = 45
SINGLE_OBJECT = True
ALLOW_OVERLAP = False

# noise generation parameters
# after which frame to add the generated noise
NOISE_AFTER_FRAME = -1

# playback parameters
PLAYBACK_SPEED = 1

def process_video(path):
    """
    Runs realtime detection on the specified video

    :param path: path to a video file to run the detection on
    """
    # ensure the provided path is file
    assert os.path.isfile(path)

    # open video capture
    cap = cv2.VideoCapture(path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print "\t Width: ", frame_width
    print "\t Height: ", frame_height
    print "\t Framerate: ", video_fps
    print "\t Number of Frames: ", num_frames

    # create gui windows to display
    #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Video + Noise', cv2.WINDOW_NORMAL)

    plt.ion()
    plt.figure()
    plt.axis([0, num_frames, 0, MAX_MOVE_DIST])

    # initialize detector
    detector = SVMDetector(Classifier('ball', 'ml/model'), SLIDING_WINDOW_SIZE)

    # initialize tracker
    tracker = Tracker(max_move_dist=MAX_MOVE_DIST,
                      max_skip_frames=MAX_SKIP_FRAMES)

    # open video capture if not open already
    if not cap.isOpened():
        cap.open(path)

    # process video frame by frame while video capture is open
    prev_detected_rois = []
    while cap.isOpened():

        # get next frame
        ret, frame_org = cap.read()

        #check for successful frame read
        if not ret:
            #failed to read frame, reset and loop the video capture back
            tracker.reset()
            cap.set(cv2.CAP_PROP_POS_FRAMES, tracker.frame_index)
            plt.clf()
            plt.axis([0, num_frames, 0, MAX_MOVE_DIST])
            continue

        #convert original frame to grayscale
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)

        # apply noise to frame if applicable
        if NOISE_AFTER_FRAME < tracker.frame_index:
            frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='gaussian', seed=None, clip=True, var=0.001))
            #frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='localvar', seed=None, clip=True))
            #frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='salt', seed=None, clip=True))
            #frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='s&p', seed=None, clip=True))
            #frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='poisson', seed=None, clip=True))
            # frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='pepper', seed=None, clip=True))
            #frame = skimage.img_as_ubyte(skimage.util.random_noise(frame, mode='speckle', seed=None, clip=True))

        # detect objects on foreground
        detected_rois = detector.detect_objects(frame, prev_detected_rois, single_object=SINGLE_OBJECT,
                                                allow_overlap=ALLOW_OVERLAP)
        tracker.frame_update_using_rois(detected_rois)
        prev_detected_rois = detected_rois

        # draw the objects onto the frame
        dist_errs, _ = tracker.mark_tracking(frame)

        # Display the resulting frame
        cv2.imshow('Video + Noise', frame)

        # update error plot
        if len(dist_errs) > 0:
            print 'Frame: ', tracker.frame_index, 'Distance_error: ', dist_errs[0]
            plt.scatter(tracker.frame_index, dist_errs[0])
        else:
            print 'Frame: ', tracker.frame_index
        plt.show()
        plt.pause(0.0001)

        # stop processing if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # delay to limit video display speed to it's FPS
        sleep(1.0 / (video_fps * PLAYBACK_SPEED))

    # done processing, release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) == 1:
        print "Not enough arguments\nUsage: python dve.py <path to a video file>"
        return
    else:
        path = argv[1]
        assert os.path.isfile(path)

        filename = os.path.basename(path)
        print "\nProcessing {} ...".format(filename)
        process_video(path)


if __name__ == '__main__':
    main(sys.argv)
