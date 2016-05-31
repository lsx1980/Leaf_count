"""
Version: 1.0
Summary: Counting leaves based on mask filtering and distance map based watershed segmentation
Author: suxing liu
Author-email: suxingliu@gmail.com

USAGE
python watershed_cv.py --image_mask mask.png --sigma 2 --min_distance 40

python watershed_cv.py --image_mask optimized_2.19.16_100_0_mask.jpg --sigma 2 --min_distance 65


"""
# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage import feature, measure
from skimage.filter import canny

from matplotlib.colors import ListedColormap
from skimage import morphology


def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_mask", required=True,	help="path to input image_mask")
ap.add_argument("-s", "--sigma", required=True,	type = int, help="gaussian_filter parameter")
ap.add_argument("-d", "--min_distance", required=True,	type = int, help="min distance in peak_local_max function")
args = vars(ap.parse_args())

# load the mask image_mask and perform pyramid mean shift filtering
image_mask = cv2.imread(args["image_mask"])

# apply meanshif filering
shifted = cv2.pyrMeanShiftFiltering(image_mask, 21, 51)
shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

# apply gaussican filtering to smooth the edges
blurred = gaussian_filter(shifted_gray, sigma = args["sigma"])


# convert the mean shift image_mask to grayscale, then apply Otsu's thresholding
mask_output = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

from skimage import morphology
from skimage.morphology import disk
#from skimage.segmentation import clear_border

# remove artifacts connected to image border
#cleared = mask_output.copy()
#im_bw_cleared = clear_border(cleared)

im_bw_cleared = morphology.remove_small_objects(mask_output, min_size = 350)

#im_bw_cleared = morphology.remove_small_objects(im_bw_cleared, 1000, connectivity = 2)
	
#remove small holes and objects 
from scipy import ndimage as ndi
label_objects, num_labels = ndi.label(im_bw_cleared)
#print num_labels
sizes = np.bincount(label_objects.ravel()) 
mask_sizes = sizes > 2000
mask_sizes[0] = 0
img_cleaned = mask_sizes[label_objects]

mask_output = img_cleaned 


# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map

thresh = mask_output

D = ndimage.distance_transform_edt(thresh)

localMax = peak_local_max(D, indices = False, min_distance = args["min_distance"], labels = thresh)

edges = canny(blurred, sigma = 2)

#save refined mask result
fig = plt.figure(frameon=False)
DPI = fig.get_dpi()
fig.set_size_inches(2454.0/float(DPI),2056.0/float(DPI))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(thresh, cmap='gray', interpolation='none')
fig_name = ('leafcount' + '_' + str(args["image_mask"][0:-4]) + '_MaskRefine.png')
fig.savefig(fig_name, dpi = DPI)

#save refined mask result
fig = plt.figure(frameon=False)
DPI = fig.get_dpi()
fig.set_size_inches(2454.0/float(DPI),2056.0/float(DPI))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(D, cmap='gray', interpolation='none')
fig_name = ('leafcount' + '_' + str(args["image_mask"][0:-4]) + '_DistanceMap.png')
imshow_overlay( D, edges, alpha=1, cmap='gray')
fig.savefig(fig_name, dpi = DPI)

#show mask comparision results in one figure
"""
#plt.figure(1)
fig, axes = plt.subplots(ncols = 3)
ax0, ax1, ax2 = axes

ax0.imshow(image_mask, cmap='gray', interpolation='none')
ax0.set_title('Input mask image_mask')

ax1.imshow(thresh, cmap='gray', interpolation='none')
ax1.set_title('Output mask image_mask')

#ax2.imshow(D, cmap='gray', interpolation='none')
imshow_overlay( D, edges, alpha=1, cmap='gray')
ax2.set_title('Distance map image_mask')

for ax in axes:
    ax.axis('off')
fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

fig = plt.gcf()
DPI = fig.get_dpi()
fig.set_size_inches(1518.0/float(DPI),1207.0/float(DPI))

#save result image
fig_name = ('leafcount' + '_' + str(args["image_mask"][0:-4]) + '.png')
fig.savefig(fig_name,dpi=DPI)
"""

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

#markers = measure.label(localMax)

labels = watershed(-D, markers, mask=thresh)

#labels = morphology.watershed(-D, markers, mask=thresh)


print("[Leaf Counting] {} unique leaves found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(shifted_gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	#((x, y), r) = cv2.minEnclosingCircle(c)
	#cv2.circle(labels, (int(x), int(y)), int(r), (0, 255, 0), 2)
	#cv2.putText(labels, "{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# show the output image_mask
plt.figure()
plt.imshow(labels, cmap="spectral", interpolation='none')
plt.title('Leaf Counting Results:' + ' Number of leaves = ' + str(len(np.unique(labels)) - 1)
+ '\n sigma = ' + str(args["sigma"]) + '    min_distance = ' + str(args["min_distance"]))
plt.axis('off')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

fig = plt.gcf()
DPI = fig.get_dpi()
#fig.set_size_inches(1518.0/float(DPI),1207.0/float(DPI))
fig.set_size_inches(2454.0/float(DPI),2056.0/float(DPI))

#save result image
fig_name = ('leafcount' + '_' + str(args["image_mask"][0:-4]) + '_parameters.png')
fig.savefig(fig_name,dpi=DPI)


#save labeled result
fig = plt.figure(frameon=False)
DPI = fig.get_dpi()
fig.set_size_inches(2454.0/float(DPI),2056.0/float(DPI))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(labels, cmap="spectral", interpolation='none')
fig_name = ('leafcount' + '_' + str(args["image_mask"][0:-4]) + '_result.png')
fig.savefig(fig_name, dpi = DPI)

#display the results
#plt.show()
#plt.close(fig)


