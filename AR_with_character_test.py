import numpy as np
import cv2 as cv

# The given video and calibration data
input_file = "20240331_185616.mp4"
K = np.array(
    [
        [967.46860897, 0, 939.92739218],
        [0, 982.20345148, 535.37527144],
        [0, 0, 1],
    ]
)
dist_coeff = np.array(
    [
        -0.00356998,
        0.01703314,
        0.00016152,
        -0.00203757,
        -0.01620528,
    ]
)
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = (
    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
)

# Open a video
video = cv.VideoCapture(input_file)
assert video.isOpened(), "Cannot read the given input, " + input_file

# Prepare a 3D box for simple AR
circle_upper = board_cellsize * np.array(
    [
        [3, 0, -2],
        [4, 0, -2],
        [5, 0.5, -2],
        [5.5, 1, -2],
        [6, 2, -2],
        [6, 3, -2],
        [5.5, 4, -2],
        [5, 4.5, -2],
        [4, 5, -2],
        [3, 5, -2],
        [2, 4.5, -2],
        [1.5, 4, -2],
        [1, 3, -2],
        [1, 2, -2],
        [1.5, 1, -2],
        [2, 0.5, -2],
    ]
)
left_eye = board_cellsize * np.array([[2, 1.5, -2], [3, 2, -2], [2, 2.5, -2]])
right_eye = board_cellsize * np.array([[5, 1.5, -2], [4, 2, -2], [5, 2.5, -2]])
mouth = board_cellsize * np.array([[2.5, 3.5, -2], [4.5, 3.5, -2], [3.5, 4.5, -2]])
nose_left = board_cellsize * np.array([[3, 3, -2], [3.5, 2.5, -2], [3.5, 3, -3]])
nose_right = board_cellsize * np.array([[4, 3, -2], [3.5, 2.5, -2], [3.5, 3, -3]])
nose_bottom = board_cellsize * np.array([[3, 3, -2], [4, 3, -2], [3.5, 3, -3]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
)

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    complete, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw on the image
        circle, _ = cv.projectPoints(circle_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(circle)], True, (0, 0, 0), 4)
        eye_left, _ = cv.projectPoints(left_eye, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(eye_left)], True, (0, 0, 255), 4)
        eye_right, _ = cv.projectPoints(right_eye, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(eye_right)], True, (0, 0, 255), 4)
        mouth_line, _ = cv.projectPoints(mouth, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(mouth_line)], True, (0, 0, 255), 4)
        nose_left_line, _ = cv.projectPoints(nose_left, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(nose_left_line)], True, (255, 0, 0), 4)
        nose_right_line, _ = cv.projectPoints(nose_right, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(nose_right_line)], True, (255, 0, 0), 4)
        nose_bottom_line, _ = cv.projectPoints(nose_bottom, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(nose_bottom_line)], True, (255, 0, 0), 4)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)  # Alternative) scipy.spatial.transform.Rotation
        p = (-R.T @ tvec).flatten()
        info = f"XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow("Pose Estimation (Chessboard)", img)
    key = cv.waitKey(10)
    if key == ord(" "):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
