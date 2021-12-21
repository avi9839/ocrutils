import numpy as np
import cv2
from scipy.spatial import distance as dist

class DocOrientationRefinement:
    """
    Document image orientation correction.
    This approach is based on text orientation.
    Assumption: Document image contains all text in same orientation.
    """
    def __init__(self):
        pass

    def refineOrientation(self, img):
        # img = cv2.imread(filePath)
        textImg = img.copy()

        small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

        # find the gradient map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

        # display(grad)

        # Binarize the gradient image
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # display(bw)

        # connect horizontally oriented regions
        # kernal value (9,1) can be changed to improved the text detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        # display(connected)

        # using RETR_EXTERNAL instead of RETR_CCOMP
        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)
        # display(mask)
        # cumulative theta value
        cummTheta = 0
        # number of detected text regions
        ct = 0
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y + h, x:x + w] = 0
            # fill the contour
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            # display(mask)
            # ratio of non-zero pixels in the filled region
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

            # assume at least 45% of the area is filled if it contains text
            if r > 0.45 and w > 8 and h > 8:
                # cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

                rect = cv2.minAreaRect(contours[idx])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(textImg, [box], 0, (0, 0, 255), 2)

                # we can filter theta as outlier based on other theta values
                # this will help in excluding the rare text region with different orientation from ususla value
                box = self.order_points(box)
                theta = self.slope(box[0][0], box[0][1], box[1][0], box[1][1])
                cummTheta += theta
                ct += 1

        # find the average of all cumulative theta value
        orientation = (cummTheta / ct)
        # print(ct)
        print("Image orientation in degress: ", orientation)
        finalImage = self.rotate(img, orientation)
        # display(textImg, "Detectd Text minimum bounding box")
        # display(finalImage, "Deskewed Image")

        return finalImage

    # rotate the image with given theta value
    def rotate(self, img, theta):
        # theta = -0.0256
        rows, cols = img.shape[0], img.shape[1]
        image_center = (cols / 2, rows / 2)

        M = cv2.getRotationMatrix2D(image_center, theta, 1)

        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])

        bound_w = int(rows * abs_sin + cols * abs_cos)
        bound_h = int(rows * abs_cos + cols * abs_sin)

        M[0, 2] += bound_w / 2 - image_center[0]
        M[1, 2] += bound_h / 2 - image_center[1]

        # rotate orignal image to show transformation
        rotated = cv2.warpAffine(img, M, (bound_w, bound_h), borderValue=(255, 255, 255))
        return rotated

    def slope(self, x1, y1, x2, y2):
        if x1 == x2:
            return 0
        slope = (y2 - y1) / (x2 - x1)
        theta = np.rad2deg(np.arctan(slope))
        return theta

    def order_points(self, pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")