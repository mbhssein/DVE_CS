
# Training SVM model for a new object

cd ml

python save_frames.py ../samples/sample2.m4v

# manually crop the target object from the frame images
# and move cropped images to new directory. i.e. samples/ball

python create_samples.py samples/ball

python train_svm.py ball samples/ball/samples.csv model


# Using the detector

cd <project directory>

python cs_tracker.py samples/sample2.m4v
