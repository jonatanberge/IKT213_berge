import numpy as np
import cv2


#task i
def sobel_edge_detection(image):
    dst = cv2.GaussianBlur(image, (3, 3), 0)
    sobelxy = cv2.Sobel(dst, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    cv2.imshow("Sobel Edge Detection", sobelxy)
    cv2.imwrite("sobel_lambo.png", sobelxy)
    return sobelxy

#task ii
def canny_edge_detection(image, threshold_1, threshold_2):
    dst = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(dst, threshold_1, threshold_2)
    cv2.imwrite("canny_lambo.png", edges)
    return edges

#task iii
def template_match(image, template):
    gray_shape = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = gray_template.shape[::-1]
    result = cv2.matchTemplate(gray_template, gray_shape, cv2.TM_CCOEFF_NORMED)

    treshold = 0.9
    loc = np.where(result >= treshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("template_match.png", image)
    return image

#task iv
def resize(image, scale_factor: int, up_or_down: str):


    if up_or_down == "up":
        for _ in range(scale_factor - 1):
            image = cv2.pyrUp(image)

    elif up_or_down == "down":
        for _ in range(scale_factor - 1):
            image = cv2.pyrDown(image)


    cv2.imwrite("resize.png", image)
    return image


image = cv2.imread('lambo.png')
shapes_image = cv2.imread('shapes-1.png')
match_image = cv2.imread('shapes_template.jpg')


edge_img = sobel_edge_detection(image)

canny_img = canny_edge_detection(image, 50, 50)

template = template_match(shapes_image, match_image)


scale = int(input("Please enter scale factor: "))
up_or_down = str(input("Please enter up or down: ")).strip().lower()

resized = resize(image, scale, up_or_down)


cv2.destroyAllWindows()