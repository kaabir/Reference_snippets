# Finding contours of the mask
# https://forum.image.sc/t/is-it-possible-to-get-coordinates-for-segmentation-mask-generated-by-felzenszwalb-functions/78544/4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import morphology, color, filters, measure
import imageio.v3 as iio

skimage.measure.find_contours to find the coordinates around the walls:

thresholded_min = closed < filters.threshold_minimum(closed)
contours = measure.find_contours(thresholded_min)

fig, ax = plt.subplots()
ax.imshow(gray, cmap='gray')
ax.set_axis_off()

for contour in contours:
    contour_xy = contour[:, [1, 0]]  # matplotlib uses xy coordinates, not row/col
    ax.add_patch(patches.Polygon(
            contour_xy,
            facecolor=(0, 0, 0, 0),
            edgecolor='cornflowerblue',
            linewdith=1))
