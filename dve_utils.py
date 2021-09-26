import math
from collections import namedtuple

import cv2
import numpy as np

import cs_utils
from ml import Classifier

# Region of interest structure template for a region at (x, y) with width 'w' and height 'h'
Roi = namedtuple('Roi', 'x, y, w, h')


class SVMDetector:
    """
        Detects the contours of objects moving on the foreground.

        Call detect_objects(frame, roi_list) with each video frame.
        Keep last detected roi_list for use in next frame
    """

    def __init__(self, classifier, window_width):
        """
        Constructor.
        :param classifier: Classifier instance for the target object
        :type classifier: Classifier
        :param min_window_width: min width of the detectable object bounding square in pixels
        :param max_window_width: max width of the detectable object bounding square in pixels
        """
        self.classifier = classifier
        self.window_width = window_width
        self.step_size = 5

    def detect_objects(self, frame, roi_list, single_object=False, allow_overlap=False):
        """
         Detects the target objects on the scene starting at ROIs.
         This function must be called with each video frame.

        :param frame: an image corresponding to a video frame to process
        :param smart_scan: if set to true sliding window scan will spiral out from
        the last detected roi location until a detection is found

        :return: list of bounding rectangles corresponding to detected objects
        """

        detected_objects = []

        self.subregion_sliding_window(frame, 0, frame.shape[1] - self.window_width, 0, frame.shape[0] - self.window_width,
                                      self.window_width, detected_objects)

        return detected_objects


    def subregion_sliding_window(self, image, from_x, to_x, from_y, to_y, window_width, detected_objects, single_object=False, allow_overlap=False):
        """
        Runs a sliding window detection on a defined subregion, and appends all detected rois to detected_objects list

        :param image:
        :param from_x:
        :param to_x:
        :param from_y:
        :param to_y:
        :param window_width:
        :param detected_objects:
        :param single_object: whether to stop further detection after first detected object
        :param allow_overlap: whether to allow overlapped object detection

        """
        for y in xrange(from_y, to_y +1, self.step_size):
            for x in xrange(from_x, to_x+1, self.step_size):
                # get a window and evaluate
                test_roi = Roi(x, y, window_width, window_width)
                if self.contains_target_object(image[y:y + window_width, x:x + window_width], detected_objects)\
                        and (not allow_overlap and not self.intercencts_any(test_roi, detected_objects)):
                    detected_objects.append(test_roi)
                    if single_object:
                        return



    def contains_target_object(self, window_image, detected_rois):
        # scale for evaluation
        image = cv2.resize(window_image, (20, 20))

        # get sparse representation
        coefficients = cs_utils.fourier_coeffs(image, 100)

        # evaluate classify the sample
        samples = np.array(coefficients, dtype=np.complex128).reshape(1, len(coefficients))
        prediction_set = self.classifier.classify(samples)

        return prediction_set[0]


    def intercencts_any(self, roi, roi_list):
        """
        Checks whether a ROI intersects any other roi in the list
        :param roi: roi to check for intersection in the list
        :param roi_list: list of rois detected so far
        :return: True if the given ROI intersects any other ROI in the list, False otherwise
        :rtype bool
        """

        for other_roi in roi_list:
            if self.intersects(roi, other_roi):
                return True
        return False


    def intersects(self, roi1, roi2):
        """
        Checks whether two regions of interest overlap.

        :param roi1: first region of interest
        :type roi1: namedtuple
        :param roi2: second region of interest
        :type roi2: namedtuple
        :return: True if the two ROIs share any point in 2D space, False otherwise
        :rtype: bool
        """

        if roi1.x <= roi2.x + roi2.w and roi1.x + roi1.w >= roi2.x and roi1.y <= roi2.y + roi2.h and roi1.y + roi1.h >= roi2.y:
            return True
        return False

class Tracker:
    """
    Simultaneously tracks multiple detected objects.
    """

    def __init__(self, max_move_dist, max_skip_frames):
        """
        Constructor

        :param max_move_dist: maximum distance in pixels an object is allowed to move to be tracked as the same object
        :param max_skip_frames: maximum number of frames an object is allowed to avoid detection to be tracked as the same object
        """
        self.max_move_dist = max_move_dist
        self.max_skip_frames = max_skip_frames
        self.frame_index = 0
        self.next_uid = 0
        self.objects = None
        self.reset()

    def reset(self):
        """
        Resets the Tracker components to process a new sequence of frames
        """
        self.frame_index = 0
        self.next_uid = 0
        self.objects = dict()

    def frame_update(self, contours):
        """
        Updates object tracking with the detection information from the current frame.

        :param contours: list of detected object contours
        """
        for cnt in contours:

            # for each contour
            # get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # bounding box center point
            mp = np.array([[np.float32(x + w / 2.0)], [np.float32(y + h / 2.0)]])

            # search for an object from previous frame which has closes predicted location to this contour
            match_id = None
            min_dist = self.max_move_dist
            for obj in self.objects.values():
                dist = Tracker.distance(mp, obj.pred_p)
                if dist < min_dist:
                    min_dist = dist
                    match_id = obj.uid

            if match_id:
                # found the matching object
                obj = self.objects[match_id]
                obj.x = x
                obj.y = y
                obj.w = w
                obj.h = h
                obj.meas_p = mp
                obj.last_frame_detected = self.frame_index
                obj.kalman.correct(mp)
            else:
                # no match was found, create new tracked object
                obj = Trackable2D(uid=self.next_uid,
                                      x=x,
                                      y=y,
                                      w=w,
                                      h=h,
                                      last_frame_detected=self.frame_index,
                                      meas_p=mp,
                                      pred_p=mp.copy(),
                                      kalman=Tracker.create_kalman2d())
                self.next_uid += 1
                self.objects[obj.uid] = obj

                # set initial position of kalman filter to the detected object center
                while True:
                    obj.kalman.correct(mp)
                    prediction = obj.kalman.predict()
                    obj.pred_p = (int(prediction[0]), int(prediction[1]))
                    if self.distance(obj.meas_p, obj.pred_p) < 1.0:
                        break

        # remove tracking of objects that have been lost or expired
        remove_ids = []
        for obj in self.objects.values():
            # check if object has not been detected for too long
            if self.frame_index - obj.last_frame_detected > self.max_skip_frames:
                remove_ids.append(obj.uid)
                continue

            prediction = obj.kalman.predict()
            obj.pred_p = (int(prediction[0]), int(prediction[1]))

        for uid in remove_ids:
            del self.objects[uid]

        self.frame_index += 1

    def frame_update_using_rois(self, roi_list):
        """
        Updates object tracking with the detection information from the current frame.

        :param roi_list: list of detected object ROIs
        :type roi_list: list
        """
        for roi in roi_list:

            # for each contour
            # get bounding rectangle
            x = roi.x
            y = roi.y
            w = roi.w
            h = roi.h

            # bounding box center point
            mp = np.array([[np.float32(x + w / 2.0)], [np.float32(y + h / 2.0)]])

            # search for an object from previous frame which has closes predicted location to this contour
            match_id = None
            min_dist = self.max_move_dist
            for obj in self.objects.values():
                dist = Tracker.distance(mp, obj.pred_p)
                if dist < min_dist:
                    min_dist = dist
                    match_id = obj.uid

            if match_id:
                # found the matching object
                obj = self.objects[match_id]
                obj.x = x
                obj.y = y
                obj.w = w
                obj.h = h
                obj.meas_p = mp
                obj.last_frame_detected = self.frame_index
                obj.kalman.correct(mp)
            else:
                # no match was found, create new tracked object
                obj = Trackable2D(uid=self.next_uid,
                                      x=x,
                                      y=y,
                                      w=w,
                                      h=h,
                                      last_frame_detected=self.frame_index,
                                      meas_p=mp,
                                      pred_p=mp.copy(),
                                      kalman=Tracker.create_kalman2d())
                self.next_uid += 1
                self.objects[obj.uid] = obj

                # set initial position of kalman filter to the detected object center
                while True:
                    obj.kalman.correct(mp)
                    prediction = obj.kalman.predict()
                    obj.pred_p = (int(prediction[0]), int(prediction[1]))
                    if self.distance(obj.meas_p, obj.pred_p) < 1.0:
                        break

        # remove tracking of objects that have been lost or expired
        remove_ids = []
        for obj in self.objects.values():
            # check if object has not been detected for too long
            if self.frame_index - obj.last_frame_detected > self.max_skip_frames:
                remove_ids.append(obj.uid)
                continue

            prediction = obj.kalman.predict()
            obj.pred_p = (int(prediction[0]), int(prediction[1]))

        for uid in remove_ids:
            del self.objects[uid]

        self.frame_index += 1


    def mark_tracking(self, frame):
        """
        Draws the detected and predicted bounding boxes onto the given image.

        :param frame: image to draw the bounding boxes on
        """
        frame_ids = []
        dist_errs = []
        for obj in self.objects.values():
            # For all objects currently being tracked
            # if objects was present in last frame draw a green detection rectangle
            if obj.last_frame_detected == self.frame_index - 1:
                #cv2.rectangle(frame, (obj.x, obj.y), (obj.x + obj.w, obj.y + obj.h), (0, 255, 0), 2)
                cv2.drawMarker(frame, (obj.x + obj.w/2, obj.y + obj.h/2), (0, 0, 0), markerType=cv2.MARKER_DIAMOND,
                               markerSize=25, line_type=cv2.LINE_8)

            #obtain top left coordinates of the predicted bounding box
            p_x = int(obj.pred_p[0] - (obj.w / 2.0))
            p_y = int(obj.pred_p[1] - (obj.h / 2.0))
            # draw the predicted bounding box
            cv2.rectangle(frame, (p_x, p_y), (p_x + obj.w, p_y + obj.h), (0, 0, 255), 1)

            #draw the id string of current object
            cv2.putText(frame, "id={}".format(obj.uid), (obj.x, obj.y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            # print detection info and error
            mp = None
            if obj.last_frame_detected == self.frame_index - 1:
                mp = (obj.meas_p[0][0], obj.meas_p[1][0])
            frame_ids.append(self.frame_index)
            dist_err = Tracker.distance(obj.meas_p, obj.pred_p)
            dist_errs.append(dist_err)
            # print "id: {}\tDetected at: {}\tPredicted at: {}\tDistance error: {:.1f}".format(obj.uid,
            #                                                                              mp,
            #                                                                              obj.pred_p,
            #                                                                               dist_err)
        return dist_errs, frame_ids


    def get_roi_list(self):
        """
        Retrieves regions of interest corresponding to objects that are being tracked.

        :return: list of ROI corresponding to objects that are being tracked.
        """
        roi_list = []
        for obj in self.objects.values():
            roi_list.append(Roi(x=obj.x, y=obj.y, w=obj.w, h=obj.h))
        return roi_list

    @staticmethod
    def distance(p0, p1):
        """
        Calculates distance between two points

        :param p0: first point in 2D space
        :param p1: second point in 2D space
        :return: distance between points p0 and p1
        """
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    @staticmethod
    def create_kalman2d():
        """
        Creates a kalman filter object for a point moving in 2D

        :return: reference to a configured cv2.KalmanFilter object
        """
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        return kalman


class Trackable2D:
    """
    Represents a trackable object.
    """
    def __init__(self, uid, x, y, w, h, last_frame_detected, meas_p, pred_p, kalman):
        """
        Constructor

        :param uid: unique id of this object
        :param x: top left x coordinate of the object bounding box
        :param y: top left y coordinate of the object bounding box
        :param w: width of the object bounding box
        :param h: height of the object bounding box
        :param last_frame_detected: the index of the last frame this object was detected on
        :param meas_p: center point of the detected bounding box
        :param pred_p: center point of the predicted bounding box
        :param kalman: 2D kalman filter assigned to this object
        """
        self.uid = uid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.last_frame_detected = last_frame_detected
        self.meas_p = meas_p
        self.pred_p = pred_p
        self.kalman = kalman
