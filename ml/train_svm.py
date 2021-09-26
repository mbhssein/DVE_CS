import os
import sys

import numpy as np
from sklearn import svm, preprocessing
from sklearn.externals import joblib

MODELS_DIR = './model'

def train_model(data_file, models_dir, model_name):

    # load all data sets
    print 'Loading training data...'

    data = np.genfromtxt(data_file, dtype=np.complex128, delimiter=',', skip_header=True)

    # create training, test and cross validation sets
    X_set = data
    num_samples = X_set.shape[0]
    test_set_bound = int(0.8 * num_samples)
    shuffled_idx = np.random.permutation(num_samples)
    training_idx, test_idx = shuffled_idx[:test_set_bound], shuffled_idx[test_set_bound:]
    X_train, X_test = X_set[training_idx,:], X_set[test_idx,:]

    print 'Total samples: ' + str(len(X_set)) + '\n' \
          + 'Training set size: ' + str(len(X_train)) + '\n' \
          + 'Test set size: ' +str(len(X_test))

    # normalize data
    print 'Preprocessing samples...'
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    # save the scaler model
    joblib.dump(scaler, os.path.join(models_dir, model_name + '_scaler.model'))

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train one class svm classifier
    print 'Training the classifier...'
    classifier = svm.OneClassSVM(nu=0.05, kernel="rbf", verbose=False)
    classifier.fit(X_train_scaled)

    # save the trained model
    joblib.dump(classifier, os.path.join(models_dir, model_name + '_classifier.model'))

    # run predictions on test and validation sets
    print 'Running predictions...'
    y_pred_test = classifier.predict(X_test_scaled)
    n_error_test = y_pred_test[y_pred_test == -1].size

    # print model prediction accuracy
    acc_test = 100 * (X_test.shape[0] - n_error_test) / X_test.shape[0]
    print 'Test set prediction accuracy: {}\n'.format(acc_test)


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) < 4:
        print "Not enough arguments\nUsage: python train_svm.py <model name> <path to a CSV data file> <path to models directory>"
        return
    else:
        model_name = argv[1]
        data_file = argv[2]
        models_dir = argv[3]
        train_model(data_file=data_file,
                    models_dir=models_dir,
                    model_name=model_name)


if __name__ == '__main__':
    main(sys.argv)