import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def harris_corners(
    reference_image: str,
    out_name: str = "harris.png",
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_rel: float = 0.1
):

    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Could not read image '{reference_image}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray32 = np.float32(gray)

    # --- Harris detection ---
    dst = cv2.cornerHarris(gray32, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    # --- Thresholding ---
    _, dst_thresh = cv2.threshold(dst, threshold_rel * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    # --- Find centroids ---
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_thresh)

    # --- Subpixel refinement ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    refined = cv2.cornerSubPix(gray32, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # --- Draw results ---
    res = np.hstack((centroids, refined))
    res = res.astype(int)

    img[res[:, 1], res[:, 0]] = [0, 0, 255]   # Red = Harris corners
    img[res[:, 3], res[:, 2]] = [0, 255, 0]   # Green = subpixel corners

    # --- Save result ---
    out_path = Path(out_name)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".png")
    cv2.imwrite(str(out_path), img)

    print(f"[INFO] Harris corners detected and saved to {out_path}")
    return str(out_path)


MIN_MATCH_COUNT = 10

img1 = cv2.imread('harris.png', cv2.IMREAD_GRAYSCALE)  # queryImage
img2 = cv2.imread('align_this.jpg', cv2.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()
if __name__ == "__main__":
    harris_corners("reference_img.png", "harris.png")
