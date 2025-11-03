import cv2
import numpy as np
from pathlib import Path

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


# === Example run per assignment ===
if __name__ == "__main__":
    align_sift(
        image_to_align="align_this.jpg",
        reference_image="reference_img.png",
        max_features=10,
        good_match_precent=0.7,       # <- Lowe ratio threshold
        aligned_out="aligned.png",
        matches_out="matches.png"
    )
