import numpy as np
import cv2

image = cv2.imread('lena.png')


def print_image_information(image):

    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)

    print("Size:", image.size)

    print("Data type:", image.dtype)


image = cv2.imread('lena.png')

print_image_information(image)

#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()