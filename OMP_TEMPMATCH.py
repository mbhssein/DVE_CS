import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from time import time
import scipy.misc
import scipy.ndimage
from PIL import Image
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from show_with_diff import show_with_diff
from add_noise import noisy


frame = cv2.imread('150.jpg',0)
shape = np.float64(cv2.imread('shape.png',0))
template = cv2.imread('template.png',0)

w, h = template.shape[::-1]

##Store frame in variale of type float32 (gray)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('frame',gray)
gray = np.asarray(frame, dtype=np.float32)
gray/=255
face=gray
##Down sample for speed
#face = gray[::2, ::2] + gray[1::2, ::2] + gray[::2, 1::2] + gray[1::2, 1::2]
face /= 0.5
height, width = face.shape
# Distort the right half of the image
 #'gauss'     Gaussian-distributed additive noise.
 #'poisson'   Poisson-distributed noise generated from the data.
 #'s&p'       Replaces random pixels with 0 or 1.
 #'speckle'   Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
print('Distorting image...')
distorted = face.copy()
#distorted += 1* np.random.randn(height, width )##Gaussian noise
distorted = noisy('s&p',distorted)
#pic = distorted *255
#pic = pic.astype('uint8')
#cv2.imwrite('noised.png',pic)
cv2.imshow('distorted',distorted)
cv2.waitKey()


# Extract all reference patches from the left half of the image
print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(shape, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))
# #############################################################################
# Learn the dictionary from reference patches
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=200, alpha=1, n_iter=500)
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

img = reconstructions[title]*255
img = img.astype(np.uint8)
method = eval('cv2.TM_CCOEFF')
# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 10)

plt.figure(2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(method)




plt.show()
