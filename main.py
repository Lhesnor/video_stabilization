import cv2
import numpy as np
from ransac import estimate_rotation, estimate_rotation_kabsch

def main(input_path, output_path, focal_length):
    cap = cv2.VideoCapture(input_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    K = np.array([[focal_length, 0, w / 2.0],
                  [0, focal_length, h / 2.0],
                  [0, 0, 1]])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)

    H_inv_cumulative = np.eye(3)

    out.write(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None:
            matches = []
        else:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:200]

        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        if len(pts1) < 4:
            stabilized = frame
        else:
            #R, inliers = estimate_rotation(pts1, pts2, K)
            R, inliers = estimate_rotation_kabsch(pts1, pts2, K)
            H = K.dot(R).dot(np.linalg.inv(K))
            H_inv = np.linalg.inv(H)
            H_inv_cumulative = H_inv.dot(H_inv_cumulative)
            stabilized = cv2.warpPerspective(frame, H_inv_cumulative, (w, h))


        out.write(stabilized)
        prev_gray = gray
        kp1, des1 = kp2, des2

    cap.release()
    out.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stabilize a handheld video via camera rotation estimation.')
    parser.add_argument('--input',  required=True, help='Path to input .mp4 video')
    parser.add_argument('--output', required=True, help='Path to save stabilized output video')
    parser.add_argument('--focal',  type=float, default=1000.0, help='Focal length in pixels (fx=fy)')
    args = parser.parse_args()
    main(args.input, args.output, args.focal)