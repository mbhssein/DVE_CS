import cv2
import numpy as np
import os
import skimage
import sys
import timeit

from matplotlib import pyplot as plt
from time import time, sleep

from dve_utils import SVMDetector, Tracker
from ml import Classifier
from sklearn.decomposition import MiniBatchDictionaryLearning


start = timeit.default_timer()
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


def preprocess_video(path):
    assert os.path.isfile(path)

    cap = cv2.VideoCapture(path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print "\t Width: ", frame_width
    print "\t Height: ", frame_height
    print "\t Framerate: ", video_fps
    print "\t Number of Frames: ", num_frames

    # open video capture if not open already
    if not cap.isOpened():
        cap.open(path)

    data = []

    while cap.isOpened():
        # get next frame
        ret, frame = cap.read()

        if not ret:
            break

        #convert original frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = cv2.resize(frame,(0,0),  fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
        #frame_width,frame_height = frame.shape
        data.append(frame)

    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(data.shape[0], -1)
    # add noise to the data
    noisy_data = data + np.random.normal(scale=0.001, size=data.shape)
    print(data.shape)
    dico = MiniBatchDictionaryLearning(n_components=300, alpha=1, n_iter=40,
                                        fit_algorithm='cd', batch_size=100,
                                        transform_alpha=5.0)

    t0 = time()
    dico.fit(noisy_data)

    print('Fit done in %.2fs.' % (time() - t0))

    # get matrix with dictionary elements
    V = dico.components_
    np.savetxt("foo.csv", V, delimiter=",")
    """
    # scale and visualize components
    for component in range(10):
        aux = V[component, :].copy()
        aux -= aux.min()
        aux /= aux.max()
        cv2.namedWindow('Component %d' % component, cv2.WINDOW_NORMAL)
        cv2.imshow('Component %d' % component, aux.reshape(frame_height, frame_width))

        while True:
            c = cv2.waitKey(1)
            if 'q' == chr(c & 255):
                cv2.destroyWindow('Component %d' % component)
                break
    """

    sum = 0

    for pos in range(10):
        frame = data[pos, :].reshape(1, -1)
        noisy_frame = noisy_data[pos, :].reshape(1, -1)
        code = dico.transform(noisy_frame)
        restored = np.dot(code, V)

        frame = frame.reshape(frame_height, frame_width)
        noisy_frame = noisy_frame.reshape(frame_height, frame_width)
        restored = restored.reshape(frame_height, frame_width)

        difference = frame - restored
        error = np.sqrt(np.sum(difference ** 2))
        sum += error
        #print('Difference (norm: %.2f)' % error)

        #restored = np.maximum(restored, 0.0)
        #restored = np.minimum(restored, 255.0)
        #restored = restored.astype(np.uint8)

        if pos < 5:
            output = np.zeros((2 * frame_height, 2 * frame_width))
            output[:frame_height, :frame_width] = frame
            output[:frame_height, frame_width:] = noisy_frame
            output[frame_height:, :frame_width] = restored
            output[frame_height:, frame_width:] = difference

            window_name = 'original, noisy / restored, difference %d' % pos
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, output / 255)

            while True:
                c = cv2.waitKey(1)
                if 'q' == chr(c & 255):
                    cv2.destroyWindow(window_name)
                    break

    sum /= 10
    print('Mean difference: %.3f' % sum)

    cap.release()

    code = dico.transform(noisy_data)
    restored = np.dot(code, V)
    print(restored.shape)
    restored = restored.reshape(-1, frame_height, frame_width)

    restored = np.maximum(restored, 0.0)
    restored = np.minimum(restored, 255.0)
    restored = restored.astype(np.uint8)

    return restored


def process_video(path, restored_frames):
    """
    Runs realtime detection on the specified video

    :param path: path to a video file to run the detection on
    """
    assert os.path.isfile(path)

    cap = cv2.VideoCapture(path)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # process video frame by frame while video capture is open
    prev_detected_rois = []

    for frame in restored_frames:
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
            plt.plot(tracker.frame_index, dist_errs[0])
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

        print "\nPreprocessing (Dictionary learning + Orthogonal Matching Pursuit) {} ...".format(filename)
        restored_frames = preprocess_video(path)

        print "\nProcessing {} ...".format(filename)
        process_video(path, restored_frames)


if __name__ == '__main__':
    main(sys.argv)

stop = timeit.default_timer()

print('Time: ', stop - start)  
