import numpy as np
import cv2
import random

def estimate_rotation(pts1, pts2, K, iterations=1000, threshold=3.0):
    num_points = pts1.shape[0]
    best_inliers = np.zeros(num_points, dtype=bool)
    best_inlier_count = 0
    best_R = np.eye(3)

    for _ in range(iterations):
        if num_points < 4:
            break
        idx = random.sample(range(num_points), 4)
        src = pts1[idx]
        dst = pts2[idx]

        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            continue
        _, Rs, _, _ = cv2.decomposeHomographyMat(H, K)
        for R in Rs:
            Hc = K.dot(R).dot(np.linalg.inv(K))
            ones = np.ones((num_points, 1))
            pts1_h = np.hstack([pts1, ones])
            proj = (Hc.dot(pts1_h.T)).T  
            proj = proj[:, :2] / proj[:, 2:3]
            error = np.linalg.norm(proj - pts2, axis=1)
            inliers = error < threshold
            inlier_count = int(np.sum(inliers))
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_R = R

    return best_R, best_inliers
