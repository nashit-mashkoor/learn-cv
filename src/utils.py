import cv2
import numpy as np

def get_opencv_img_from_buffer(buffer, flags=cv2.IMREAD_COLOR):
       # read image as an numpy array
        image = np.asarray(bytearray(buffer.read()), dtype="uint8")
        # use imdecode function
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (350, 350), cv2.INTER_AREA)
        return image
