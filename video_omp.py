import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from time import time
import scipy.misc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


def show_with_diff(image, reference, title):
    """Helper function to display denoising"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)
    plt.savefig('omp_frames/{}.png'.format(framenum))
    #scipy.misc.imsave('omp_frames/simpleframe_{}.png'.format(framenum), image)

##import video##

cap = cv2.VideoCapture('samples/sam1.mp4')

framenum = 0
while(cap.isOpened()):
    framenum += 1
    ret, frame = cap.read()
    if ret:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      #cv2.imshow('frame',gray)
      gray = np.asarray(gray, dtype=np.float32)
      gray/=255
      face=gray
    # downsample for higher speed
      #face = gray[::2, ::2] + gray[1::2, ::2] + gray[::2, 1::2] + gray[1::2, 1::2]
      #face /= 4.0
      height, width = face.shape

    # Distort the right half of the image
      print('Distorting image...')
      distorted = face.copy()
      distorted += 2* np.random.randn(height, width )

# Extract all reference patches from the left half of the image
      print('Extracting reference patches...')
      t0 = time()
      patch_size = (7, 7)
      data = extract_patches_2d(gray, patch_size)
      data = data.reshape(data.shape[0], -1)
      data -= np.mean(data, axis=0)
      data /= np.std(data, axis=0)
      print('done in %.2fs.' % (time() - t0))
      print ('frame number =', framenum)

# #############################################################################
# Learn the dictionary from reference patches

      print('Learning the dictionary...')
      t0 = time()
      dico = MiniBatchDictionaryLearning(n_components=100, alpha=10, n_iter=500)
      V = dico.fit(data).components_
      dt = time() - t0
      print('done in %.2fs.' % dt)

      plt.figure(figsize=(4.2, 4))
      for i, comp in enumerate(V[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


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

    #plt.show()




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

#show_with_diff(distorted, face, 'Distorted image')

