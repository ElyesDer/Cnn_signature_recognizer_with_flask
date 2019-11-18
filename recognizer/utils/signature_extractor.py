"""Extract signatures from an image."""
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 17th September 2018
# ----------------------------------------------

import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
# from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
from scipy import misc
import os


# added another module ; image resize

def extract_signature(source_image, identifier):
    """Extract signature from an input image.

    Parameters
    ----------
    source_image : numpy ndarray
        The pinut image.

    Returns
    -------
    numpy ndarray
        An image with the extracted signatures.

    """
    # os.mkdir("outputs/boom", 0o755)
    # read the input image
    img = source_image
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    # image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    '''
    # plot the connected components (for debugging)
    ax.imshow(image_label_overlay)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    '''

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area / counter)
    print("the_biggest_component: " + str(the_biggest_component))
    print("average: " + str(average))

    # experimental-based ratio calculation, modify it for your cases
    # a4_constant is used as a threshold value to remove connected pixels
    # are smaller than a4_constant for A4 size scanned documents
    a4_constant = (((average / 84.0) * 250.0) + 100) * 1.5
    print("a4_constant: " + str(a4_constant))

    # remove the connected pixels are smaller than a4_constant
    b = morphology.remove_small_objects(blobs_labels, a4_constant)
    # save the the pre-version which is the image is labelled with colors
    # as considering connected components
    print("Saving here :", "outputs/{0}/extracted/pre_version.png".format(identifier))

    plt.imsave("outputs/{0}/extracted/pre_version.png".format(identifier), b)
    #cv2.imwrite(os.path.join("outputs/{0}/extracted/", "pre_version.png"), b)

    #misc.imsave("pre_version.png".format(identifier), b)

    # read the pre-version
    img = cv2.imread("outputs/{0}/extracted/pre_version.png".format(identifier), 0)
    # ensure binary
    img = cv2.threshold(img, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # save the the result
    # cv2.imwrite("outputs.png", img)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and filter for ROI
    image_number = 0
    image_array = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ROI = img[y:y + h, x:x + w]

        ROI = cv2.resize(ROI, (96, 96), interpolation=cv2.INTER_AREA)

        cv2.imwrite("outputs/{0}/extracted/ROI_{1}.png".format(identifier, image_number), ROI)

        image_number += 1
        image_array.append(ROI)

    print("Channel info : ", len(img.shape))
    return img, image_array
