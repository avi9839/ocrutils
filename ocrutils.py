import cv2
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage import img_as_ubyte
from PIL import Image
import pytesseract
import numpy as np
from scipy.spatial import distance as dist

DEBUG = True

#Display image
def display(img, frameName="OpenCV Image"):
    if not DEBUG:
        return
    h, w = img.shape[0:2]
    neww = 1300
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)


def bradley_roth_numpy(img, s=None, t=None):

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


def sauvola(gray):
    window_size = 25
    thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    binary_sauvola = gray > thresh_sauvola
    th = img_as_ubyte(binary_sauvola)
    return th

def niblack(gray):
    window_size = 25
    thresh_niblack = threshold_niblack(gray, window_size=window_size, k=0.8)
    binary_niblack = gray > thresh_niblack
    th = img_as_ubyte(binary_niblack)
    return th

def adaptive(gray):
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th


#convert the raw tesseract text data to line data
def parseLines(text):

    data = {}
    for i in range(len(text['line_num'])):
        if not (text['text'][i] == '' or text['text'][i].isspace()):
            if text['block_num'][i] in data:

                if text['line_num'][i] in data[text['block_num'][i]]:
                    data[text['block_num'][i]][text['line_num'][i]].append((text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i]))
                else:
                    # lastLineKey = text['line_num'][i]
                    # line[text['line_num'][i]] = []
                    data[text['block_num'][i]][text['line_num'][i]] = [(text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i])]
                    # line[lastLineKey].append()

            else:
                data[text['block_num'][i]] = {}
                data[text['block_num'][i]][text['line_num'][i]] = [(text['text'][i], text['left'][i], text['top'][i], text['width'][i], text['height'][i])]

    linedata = {}
    idx = 0
    for _, b  in data.items():
        for _, l in b.items():
            linedata[idx] = l
            idx += 1

    return linedata

#remove lines and grids
def removeNoise(thresh):
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    thresh = 255 - thresh
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # thresh = np.zeros((thresh.shape))
    labelMat = output[1]
    stats = output[2]
    labels = output[0]
    for label in range(labels):
        w = stats[label][2]
        h = stats[label][3]
        dimRatio = w / h
        # print(dimRatio)
        pixCt = stats[label][4]
        pixDensity = pixCt / (w * h)
        # remove the table/grid shaped components
        if pixDensity < 0.1 or pixCt < 4:
            # displayImage(thresh)
            # print('Delete', dimRatio, pixDensity)
            thresh[labelMat == label] = 0

        # remove lines and small components
        if ((dimRatio < 0.05 or dimRatio > 20) and (w > thresh.shape[1]/2 or h > thresh.shape[0]/2)):
            # displayImage(thresh)
            # print('Delete', dimRatio, pixDensity)
            thresh[labelMat == label] = 0

    thresh = 255 - thresh
    # display(thresh)
    return thresh

def otsuThreshold(gray):
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


#Document image orientation correction
#This approach is based on text orientation

#Assumption: Document image contains all text in same orientation

#rotate the image with given theta value
def rotate(img, theta):
    #theta = -0.0256
    rows, cols = img.shape[0], img.shape[1]
    image_center = (cols/2, rows/2)

    M = cv2.getRotationMatrix2D(image_center,theta,1)

    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])

    bound_w = int(rows * abs_sin + cols * abs_cos)
    bound_h = int(rows * abs_cos + cols * abs_sin)

    M[0, 2] += bound_w/2 - image_center[0]
    M[1, 2] += bound_h/2 - image_center[1]

    # rotate orignal image to show transformation
    rotated = cv2.warpAffine(img,M,(bound_w,bound_h),borderValue=(255,255,255))
    return rotated


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return 0
    slope = (y2-y1)/(x2-x1)
    theta = np.rad2deg(np.arctan(slope))
    return theta


def order_points(pts):
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



def refineOrientation(img):
    #img = cv2.imread(filePath)
    textImg = img.copy()

    small = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)

    #find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    #display(grad)

    #Binarize the gradient image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #display(bw)

    #connect horizontally oriented regions
    #kernal value (9,1) can be changed to improved the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    #display(connected)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    #display(mask)
    #cumulative theta value
    cummTheta = 0
    #number of detected text regions
    ct = 0
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        #fill the contour
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        #display(mask)
        #ratio of non-zero pixels in the filled region
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        #assume at least 45% of the area is filled if it contains text
        if r > 0.45 and w > 8 and h > 8:
            #cv2.rectangle(textImg, (x1, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            rect = cv2.minAreaRect(contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(textImg,[box],0,(0,0,255),2)

            #we can filter theta as outlier based on other theta values
            #this will help in excluding the rare text region with different orientation from ususla value
            box = order_points(box)
            theta = slope(box[0][0], box[0][1], box[1][0], box[1][1])
            cummTheta += theta
            ct +=1

    #find the average of all cumulative theta value
    orientation = (cummTheta/ct)
    #print(ct)
    print("Image orientation in degress: ", orientation)
    finalImage = rotate(img, orientation)
    #display(textImg, "Detectd Text minimum bounding box")
    #display(finalImage, "Deskewed Image")

    return finalImage


def main(filePath):
    print("Main function invoked!")
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Otsu's thresholding
    # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # display(th2)
    #
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # display(th3)
    #
    # th_adaptive = adaptive(gray)
    # display(th_adaptive, "Adaptive threshold")

    #th_sauvola = sauvola(gray)
    #display(th_sauvola, "Sauvola thresholding")


    th_bradely = bradley_roth_numpy(gray)
    #th_bradely = removeNoise(255-th_bradely)
    display(th_bradely)
    gray_pil = Image.fromarray(th_bradely)

    # OCR image
    config = ('-l eng --oem 1')
    text = pytesseract.image_to_data(gray_pil, config=config, output_type='dict')
    line = parseLines(text)

    for key, arr in line.items():
        a = ''
        for item in arr:
            a += item[0] + " "
        print(a)

    print(line)

    #th_niblack = niblack(gray)
    #display(th_niblack, "Ni-black threshold")


if __name__ == "__main__":
    filePath = 'doc_img.jpg'
    main(filePath)
