import cv2
import numpy as np

# ===== CONFIGURATION =====
REF_IMAGES = {
    0: "/home/pi/Bildverarbeitung/Task 2/Part ID0 empty.png",
    1: "/home/pi/Bildverarbeitung/Task 2/Part ID01 empty.png"
}
SAMPLE_RADIUS = 30
HISTORY_LENGTH = 5

aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
PARAMS = aruco.DetectorParameters_create()

MARKER_POSITIONS = {
    0: ((135, 140), 120),
    1: ((580, 420), 120)
}

# ===== REFERENCE PROCESSING =====
def get_holes(ref_path):
    ref = cv2.imread(ref_path, 0)
    if ref is None:
        print(f"[ERROR] Could not load image: {ref_path}")
        return []
    blurred = cv2.GaussianBlur(ref, (11, 11), 3)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=60,
        param2=40,
        minRadius=40,
        maxRadius=85
    )
    holes = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        holes = [(c[0], c[1]) for c in circles]
    print(f"[REFERENCE] {ref_path} → {len(holes)} holes detected")
    return holes

# Preload hole positions from reference images
ref_holes = {}
for mid, path in REF_IMAGES.items():
    ref_holes[mid] = get_holes(path)

# Homography smoothing buffers
homography_buffers = {0: [], 1: []}

# ===== CAMERA SETUP =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, DICT, parameters=PARAMS)

    if ids is not None:
        for i, id_ in enumerate(ids.flatten()):
            if id_ not in ref_holes:
                continue

            holes = ref_holes[id_]
            current_corners = corners[i][0].astype(np.float32)

            # Moving average smoothing over last HISTORY_LENGTH frames
            buffer = homography_buffers[id_]
            buffer.append(current_corners)
            if len(buffer) > HISTORY_LENGTH:
                buffer.pop(0)
            smoothed_corners = np.mean(buffer, axis=0)

            # Build source points from known marker position in reference image
            (ox, oy), marker_size = MARKER_POSITIONS[id_]
            src_pts = np.array([
                [ox,              oy],
                [ox + marker_size, oy],
                [ox + marker_size, oy + marker_size],
                [ox,              oy + marker_size]
            ], dtype=np.float32)

            # Compute homography: reference image → camera frame
            H, _ = cv2.findHomography(src_pts, smoothed_corners)
            if H is None:
                continue

            for (x, y) in holes:
                # Map hole position from reference image to camera frame
                ref_pt = np.array([[[x, y]]], dtype=np.float32)
                cam_pt = cv2.perspectiveTransform(ref_pt, H)[0][0]
                x_t, y_t = int(cam_pt[0]), int(cam_pt[1])

                # Extract region of interest around the hole
                x1 = max(x_t - SAMPLE_RADIUS, 0)
                y1 = max(y_t - SAMPLE_RADIUS, 0)
                x2 = min(x_t + SAMPLE_RADIUS, gray.shape[1] - 1)
                y2 = min(y_t + SAMPLE_RADIUS, gray.shape[0] - 1)
                roi = gray[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # Otsu thresholding to separate screw from empty hole
                roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
                _, roi_thresh = cv2.threshold(
                    roi_blur, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

                # Screw head is bright → more white pixels after inversion
                num_black = np.sum(roi_thresh == 0)
                num_white = np.sum(roi_thresh == 255)
                is_screw = num_black < num_white

                color = (0, 255, 0) if is_screw else (0, 0, 255)
                label = "SCREW" if is_screw else "HOLE"

                cv2.circle(frame, (x_t, y_t), SAMPLE_RADIUS + 5, color, 2)
                cv2.putText(
                    frame, label,
                    (x_t - 40, y_t - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )

            aruco.drawDetectedMarkers(
                frame,
                [smoothed_corners.reshape(1, 4, 2)],
                np.array([[id_]])
            )

    cv2.imshow("Screw Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
