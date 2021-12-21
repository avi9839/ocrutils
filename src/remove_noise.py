import cv2

class RemoveNoise:
    """
    Remove noise from document image which help in improving the OCR accuracy.
    """
    def __init__(self):
        pass

    def removeNoise(self, image):
        """
        Remove lines and grids

        Args:
            image: OpenCV Binary Image

        Returns:
            OpenCV image without lines and grids
        """
        if image is not None and len(image.shape) == 3:
            thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            thresh = image

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

if __name__ == "__main__":
    filePath = '../img/table_img.png'
    remove_noise = RemoveNoise()
    img = cv2.imread(filePath)
    res_img = remove_noise.removeNoise(img)
    cv2.imshow("Image Without Table", res_img)
    cv2.waitKey(0)