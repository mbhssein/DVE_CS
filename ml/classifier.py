import os

from sklearn.externals import joblib


class ObjectClassifier:
    """
    Provides access to pre-trained field classifier
    """

    def __init__(self, model_name, model_dir):
        """
        Constructor

        :param model_name: name of this model
        :type model_name: str
        :param model_dir: directory where the model files are located
        :type model_dir: str
        """
        self.model = joblib.load(os.path.join(model_dir, model_name + '_classifier.model'))
        self.scaler = joblib.load(os.path.join(model_dir, model_name + '_scaler.model'))


    def classify(self, coefficients):
        """
        Classifies the given samples returning boolean flags indicating
        whether each sample can be classified as the target object

        :param coefficients: 2D numpy array containing rows of samples to evaluate.
        Sample entries must have same number of parameters as the entries in training sample for the classifier model.
        :type coefficients: numpy.ndarray
        :return: list of booleans representing classification labels for rows of field_candidates
        :rtype: list
        """

        # object_classifier = ObjectClassifier('ball', ./model)
        # samples = np.array(sample).reshape(1, len(sample))
        # is_target_object = prediction_set[0]

        predictions = self.model.predict(self.scaler.transform(coefficients))
        return predictions == 1