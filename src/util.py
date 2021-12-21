import cv2

class Utility:
    def __init__(self):
        pass

    # Display image
    def display(self, img, wait_key=0, frame_name="OpenCV Image"):
        """
        Display OpenCV image.

        Args:
            img: OpenCV Image
            wait_key: pass 0 to display the image for unlimited time,
            pass positive integer to display the image for given miliseconds
            frame_name: window title in which image will be shown
        """
        h, w = img.shape[0:2]
        neww = 1300
        newh = int(neww * (h / w))
        img = cv2.resize(img, (neww, newh))
        cv2.imshow(frame_name, img)
        cv2.waitKey(wait_key)