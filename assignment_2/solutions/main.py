import numpy as np
import cv2


#Task i

def padding (image, border_width):
    padded_image = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)

    cv2.imwrite("lena_padded.png", padded_image)
    print(f"Padded image saved as lena_padded.png with border width {border_width}")
    return padded_image


#Task ii

def crop (image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("cropped_image.png", cropped_image)
    print(f"Cropped image saved as cropped_image.png")
    return cropped_image

#Task iii

def resize (image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("resized_image.png", resized_image)
    print(f"Resized image saved as resized_image.png")
    return resized_image


#Task iv

def copy (image, emptyPictureArray):
    height, width, channels = image.shape

    emptyPictureArray = np.zeros((512, 512, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]

    cv2.imwrite("copied_image.png", emptyPictureArray)
    print("Copied image saved as copied_image.png")
    return emptyPictureArray

#Task v

def grayscale (image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png", gray)
    print(f"Gray image saved as gray.png")
    return gray

#Task vi

def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsv)
    print(f"HSV image saved as hsv.png")
    return hsv

#Task vii

def hue_shifted (image, emptyPictureArray, hue):
    height, width, channels = image.shape

    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):

                new_value = image[y, x, c] + hue

                new_value = max(0, min(255, new_value))

                emptyPictureArray[y, x, c] = new_value


    cv2.imwrite("hue_shifted.png", emptyPictureArray)
    print("Hue-shifted image saved as hue_shifted.png")
    return emptyPictureArray

#Task viii
def smoothing (image):
    dst = cv2.GaussianBlur(image, (15, 15), 0,  cv2.BORDER_DEFAULT)
    cv2.imwrite("smoothing.png", dst)
    print("Smoothing image saved as smoothing.png")
    return dst


#Task ix

def rotation (image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        print("Only 90 and 180 is supported.")

    cv2.imwrite("rotated.png", rotated)
    print(f"Rotated image saved as rotate.png")
    return rotated





image = cv2.imread('lena.png')

padded = padding(image, 100)

h, w, _ = image.shape
cropped = crop(image, 80, w-130, 80, h-130)


resized = resize(image, 200, 200)


emptyPictureArray = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

copied = copy(image, emptyPictureArray)

gray = grayscale(image)

hsv = hsv(image)

hue_img = hue_shifted(image, emptyPictureArray, 50)

dst = smoothing(image)

angle = int(input("Please enter the rotation angle (90 or 180): "))

rotated = rotation(image, angle)


cv2.waitKey(0)
cv2.destroyAllWindows()