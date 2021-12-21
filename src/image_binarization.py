import cv2
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage import img_as_ubyte
from PIL import Image
import numpy as np

from util import Utility

class ImageBinarization(Utility):
    """
    This module contains various popular algorithms for Image Binarization.
    """
    def __init__(self):
        super().__init__()
        pass

    def bradley_roth_numpy(self, img, s=None, t=None):

        # Convert image to numpy array
        #img = np.array(image).astype(np.float)

        # Default window size is round(cols/8)
        if s is None:
            s = np.round(img.shape[1]/8)

        # Default threshold is 15% of the total
        # area in the window
        if t is None:
            t = 15.0

        # Compute integral image
        intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

        # Define grid of points
        (rows,cols) = img.shape[:2]
        (X,Y) = np.meshgrid(np.arange(cols), np.arange(rows))

        # Make into 1D grid of coordinates for easier access
        X = X.ravel()
        Y = Y.ravel()

        # Ensure s is even so that we are able to index into the image
        # properly
        s = s + np.mod(s,2)

        # Access the four corners of each neighbourhood
        x1 = X - s/2
        x2 = X + s/2
        y1 = Y - s/2
        y2 = Y + s/2

        # Ensure no coordinates are out of bounds
        x1[x1 < 0] = 0
        x2[x2 >= cols] = cols-1
        y1[y1 < 0] = 0
        y2[y2 >= rows] = rows-1

        # Ensures coordinates are integer
        x1 = x1.astype(np.int)
        x2 = x2.astype(np.int)
        y1 = y1.astype(np.int)
        y2 = y2.astype(np.int)

        # Count how many pixels are in each neighbourhood
        count = (x2 - x1) * (y2 - y1)

        # Compute the row and column coordinates to access
        # each corner of the neighbourhood for the integral image
        f1_x = x2
        f1_y = y2
        f2_x = x2
        f2_y = y1 - 1
        f2_y[f2_y < 0] = 0
        f3_x = x1-1
        f3_x[f3_x < 0] = 0
        f3_y = y2
        f4_x = f3_x
        f4_y = f2_y

        # Compute areas of each window
        sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

        # Compute thresholded image and reshape into a 2D grid
        out = np.ones(rows*cols, dtype=np.bool)
        out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

        # Also convert back to uint8
        out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

        # Return PIL image back to user
        return out


    def sauvola(self, gray):
        window_size = 25
        thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
        binary_sauvola = gray > thresh_sauvola
        th = img_as_ubyte(binary_sauvola)
        return th

    def niblack(self, gray):
        window_size = 25
        thresh_niblack = threshold_niblack(gray, window_size=window_size, k=0.8)
        binary_niblack = gray > thresh_niblack
        th = img_as_ubyte(binary_niblack)
        return th

    def adaptive(self, gray):
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return th

    def otsuThreshold(self, gray):
        ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2


if __name__ == "__main__":
    filePath = '../img/binarization_image.png'
    img = cv2.imread(filePath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Instantiate the Image Binarization Object
    image_binarization = ImageBinarization()

    bw_img = image_binarization.otsuThreshold(gray_img)
    cv2.imshow("OTSU BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.adaptive(gray_img)
    cv2.imshow("Adaptive BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.bradley_roth_numpy(gray_img)
    cv2.imshow("Bradley BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.niblack(gray_img)
    cv2.imshow("Niblack BW", bw_img)
    cv2.waitKey(0)

    bw_img = image_binarization.sauvola(gray_img)
    cv2.imshow("Sauvola BW", bw_img)
    cv2.waitKey(0)


