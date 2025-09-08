import numpy as np
import cv2

image = cv2.imread('lena.png')

# Task IV
def print_image_information(image):

    height, width, channels = image.shape

    print("Height:", height)

    print("Width:", width)

    print("Channels:", channels)

    print("Size:", image.size)

    print("Data type:", image.dtype)


image = cv2.imread('lena.png')

print_image_information(image)

# Task V

cam = cv2.VideoCapture(0)

fps = cam.get(cv2.CAP_PROP_FPS)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


with open('camera_outputs.txt', 'w') as f:
    f.write(f"Width: {frame_width}\n")
    f.write(f"Height: {frame_height}\n")
    f.write(f"FPS: {fps}\n")


cam.release()