import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from time import time
import scipy.misc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from show_with_diff import show_with_diff
from add_noise import noisy

cap = cv2.VideoCapture('samples/sample2.m4v')

framenum = 0
while(cap.isOpened()):
    framenum += 1
    ret, frame = cap.read()
    if ret:

      ##Store frame in variale of type float32 (gray)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      #cv2.imshow('frame',gray)
      gray = np.asarray(gray, dtype=np.float32)
      gray/=255
      face=gray

      ##Down sample for speed
      face = gray[::2, ::2] + gray[1::2, ::2] + gray[::2, 1::2] + gray[1::2, 1::2]
      face /= 6.5
      height, width = face.shape

      # Distort the right half of the image
       #'gauss'     Gaussian-distributed additive noise.
       #'poisson'   Poisson-distributed noise generated from the data.
       #'s&p'       Replaces random pixels with 0 or 1.
       #'speckle'   Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
      print('Distorting image...')

      distorted = face.copy()
      #distorted += 1* np.random.randn(height, width )##Gaussian noise
      distorted = noisy('gauss',distorted)
      #pic = distorted *255
      #pic = pic.astype('uint8')
      #cv2.imwrite('noised.png',pic)
      cv2.imshow('distorted',distorted)
      cv2.waitKey()


      # Extract all reference patches from the left half of the image
      print('Extracting reference patches...')
      t0 = time()
      patch_size = (7, 7)
      data = extract_patches_2d(gray, patch_size)
      data = data.reshape(data.shape[0], -1)
      data -= np.mean(data, axis=0)
      data /= np.std(data, axis=0)
      print('done in %.2fs.' % (time() - t0))

      # #############################################################################
      # Learn the dictionary from reference patches

      print('Learning the dictionary...')
      t0 = time()
      dico = MiniBatchDictionaryLearning(n_components=100, alpha=10, n_iter=500)
      V = dico.fit(data).components_
      dt = time() - t0
      print('done in %.2fs.' % dt)


      # #############################################################################
      # Extract noisy patches and reconstruct them using the dictionary

      print('Extracting noisy patches... ')
      t0 = time()
      data = extract_patches_2d(distorted, patch_size)
      data = data.reshape(data.shape[0], -1)
      intercept = np.mean(data, axis=0)
      data -= intercept
      print('done in %.2fs.' % (time() - t0))

      transform_algorithms = [
     ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1})]

      reconstructions = {}
      for title, transform_algorithm, kwargs in transform_algorithms:
         print(title + '...')
         reconstructions[title] = face.copy()
         t0 = time()
         dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
         code = dico.transform(data)
         patches = np.dot(code, V)

         patches += intercept
         patches = patches.reshape(len(data), *patch_size)
         if transform_algorithm == 'threshold':
           patches -= patches.min()
           patches /= patches.max()
         reconstructions[title] = reconstruct_from_patches_2d(
           patches, (height, width))
         dt = time() - t0
         print('done in %.2fs.' % dt)
         show_with_diff(reconstructions[title], face,
                   title + ' (time: %.1fs)' % dt)

    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
