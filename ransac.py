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

def to_bearing(pts, K):
    K_inv = np.linalg.inv(K)
    pts_h = np.hstack([pts, np.ones((len(pts),1))])      # Nx3
    dirs = (K_inv @ pts_h.T).T                          # Nx3
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms                                 # нормированные Nx3

def kabsch_3d(A, B):
    # A, B: Nx3 центровые
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    return R

def estimate_rotation_kabsch(pts1, pts2, K, iterations=500, threshold=0.01):
    N = len(pts1)
    best_R = np.eye(3)
    best_inliers = np.zeros(N, bool)
    pts1_b = to_bearing(pts1, K)
    pts2_b = to_bearing(pts2, K)

    for _ in range(iterations):
        if N < 3: break
        idx = random.sample(range(N), 3) 
        A = pts1_b[idx] - np.mean(pts1_b[idx], axis=0, keepdims=True)
        B = pts2_b[idx] - np.mean(pts2_b[idx], axis=0, keepdims=True)
        R = kabsch_3d(A, B)
        D1 = (R @ pts1_b.T).T
        errs = np.linalg.norm(D1 - pts2_b, axis=1)
        inliers = errs < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_R = R

    return best_R, best_inliers