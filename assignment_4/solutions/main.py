import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def harris_corner_detection(reference_image: str, out_name: str = "harris.png"):
    # --- Read and convert ---
    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Could not read image '{reference_image}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # --- Harris detection ---
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    # --- Threshold and mark corners ---
    img[dst > 0.01 * dst.max()] = [0, 0, 255]   # red marks

    # --- Save result ---
    out_path = Path(out_name)
    cv2.imwrite(str(out_path), img)
    print(f"[INFO] Harris corners detected and saved to {out_path}")
    return str(out_path)


def align_sift(image_to_align: str,
               reference_image: str,
               max_features: int = 10,
               good_match_precent: float = 0.7,
               aligned_out: str = "aligned.png",
               matches_out: str = "matches.png"):


    # --- Load images (color for drawing, gray for features) ---
    im1_color = cv2.imread(image_to_align, cv2.IMREAD_COLOR)     # to be warped
    im2_color = cv2.imread(reference_image, cv2.IMREAD_COLOR)    # reference frame
    if im1_color is None:
        raise FileNotFoundError(f"Could not read image_to_align: {image_to_align}")
    if im2_color is None:
        raise FileNotFoundError(f"Could not read reference_image: {reference_image}")

    im1_gray = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

    # --- SIFT keypoints & descriptors (cap features per instructions) ---
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    kp2, des2 = sift.detectAndCompute(im2_gray, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        raise RuntimeError("SIFT found no features in one or both images.")

    # --- FLANN matcher (KD-Tree) + Lowe's ratio test ---
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < good_match_precent * n.distance:
            good.append(m)
    if len(good) < 4:
        raise RuntimeError(f"Not enough good matches ({len(good)}) to compute homography (need >= 4).")

    good = sorted(good, key = lambda x: x.distance)[:max_features]
    # --- Homography (RANSAC) ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # from image_to_align
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # to reference_image
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    # --- Warp image_to_align to reference frame size ---
    h, w = im2_color.shape[:2]
    aligned = cv2.warpPerspective(im1_color, H, (w, h))

    # --- Draw matches (only inliers) and save outputs ---
    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]

    match_vis = cv2.drawMatches(
        im1_color, kp1, im2_color, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(aligned_out, aligned)
    cv2.imwrite(matches_out, match_vis)

    return str(Path(aligned_out)), str(Path(matches_out))

if __name__ == "__main__":
    harris_corner_detection("reference_img.png", "harris.png")

    align_sift(
        image_to_align="align_this.jpg",
        reference_image="reference_img.png",
        max_features=10,
        good_match_precent=0.7,
        aligned_out="aligned.png",
        matches_out="matches.png"
    )
