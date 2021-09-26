import os
import sys

import cv2

def process_video(path):
    """
    Saves frames of video as images for sample collection

    :param path: path to a video file to run the sample collection on
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


    # open video capture if not open already
    if not cap.isOpened():
        cap.open(path)

    frame_index = 0

    # process video frame by frame while video capture is open
    while cap.isOpened():

        # get next frame
        ret, frame_org = cap.read()

        #check for successful frame read
        if not ret:
            #failed to read frame, reset and loop the video capture back
           break

        #convert original frame to grayscale
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('ml/samples/frames' + str(frame_index) + '.png', frame)
        frame_index += 1

        # stop processing if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # done processing, release the capture and close all windows
    cap.release()


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) == 1:
        print "Not enough arguments\nUsage: python save_frames.py <path to a video file>"
        return
    else:
        path = argv[1]
        assert os.path.isfile(path)

        filename = os.path.basename(path)
        print "\nProcessing {} ...".format(filename)
        process_video(path)


if __name__ == '__main__':
    main(sys.argv)
