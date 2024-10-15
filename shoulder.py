import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_shoulder_angles(landmarks):
    # Shoulder flexion/extension
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    flex_ext = calculate_angle(hip, shoulder, elbow)

    # Shoulder abduction/adduction
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    abd_add = calculate_angle(right_shoulder, left_shoulder, left_wrist)

    # Shoulder rotation (approximate)
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    rotation = calculate_angle(shoulder, elbow, wrist)

    return flex_ext, abd_add, rotation

def draw_text_with_background(img, text, position, font_scale, thickness, text_color, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, position, (position[0] + text_w, position[1] + text_h), bg_color, -1)
    cv2.putText(img, text, (position[0], position[1] + text_h), font, font_scale, text_color, thickness)

def calibration_phase(cap, pose, num_frames=30):
    angles = {'flex_ext': [], 'abd_add': [], 'rotation': []}
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            flex_ext, abd_add, rotation = get_shoulder_angles(results.pose_landmarks.landmark)
            angles['flex_ext'].append(flex_ext)
            angles['abd_add'].append(abd_add)
            angles['rotation'].append(rotation)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2)
            )

        draw_text_with_background(frame, "Calibrating... Stand in neutral position", (10, 30), 0.7, 2, (0, 255, 0), (0, 0, 0))
        cv2.imshow('Shoulder Range of Motion Measurement', frame)
        cv2.waitKey(1)

    return {k: np.mean(v) for k, v in angles.items()} if all(angles.values()) else None

cap = cv2.VideoCapture(0)

max_values = {'flexion': 0, 'extension': 0, 'abduction': 0, 'adduction': 0, 'internal_rotation': 0, 'external_rotation': 0}
neutral_angles = None
calibrated = False
angle_buffer = {k: deque(maxlen=5) for k in ['flex_ext', 'abd_add', 'rotation']}

print("Stand in a neutral position and press 'c' to calibrate")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2)
        )

        if calibrated:
            flex_ext, abd_add, rotation = get_shoulder_angles(results.pose_landmarks.landmark)
            
            angle_buffer['flex_ext'].append(flex_ext - neutral_angles['flex_ext'])
            angle_buffer['abd_add'].append(abd_add - neutral_angles['abd_add'])
            angle_buffer['rotation'].append(rotation - neutral_angles['rotation'])

            smoothed_flex_ext = np.mean(angle_buffer['flex_ext'])
            smoothed_abd_add = np.mean(angle_buffer['abd_add'])
            smoothed_rotation = np.mean(angle_buffer['rotation'])

            if smoothed_flex_ext > 0:
                max_values['flexion'] = max(max_values['flexion'], smoothed_flex_ext)
            else:
                max_values['extension'] = max(max_values['extension'], abs(smoothed_flex_ext))

            if smoothed_abd_add > 0:
                max_values['abduction'] = max(max_values['abduction'], smoothed_abd_add)
            else:
                max_values['adduction'] = max(max_values['adduction'], abs(smoothed_abd_add))

            if smoothed_rotation > 0:
                max_values['external_rotation'] = max(max_values['external_rotation'], smoothed_rotation)
            else:
                max_values['internal_rotation'] = max(max_values['internal_rotation'], abs(smoothed_rotation))

            draw_text_with_background(frame, f"Flexion/Extension: {smoothed_flex_ext:.2f}", (10, 30), 0.7, 2, (0, 255, 0), (0, 0, 0))
            draw_text_with_background(frame, f"Abduction/Adduction: {smoothed_abd_add:.2f}", (10, 60), 0.7, 2, (0, 255, 0), (0, 0, 0))
            draw_text_with_background(frame, f"Rotation: {smoothed_rotation:.2f}", (10, 90), 0.7, 2, (0, 255, 0), (0, 0, 0))

            y_offset = 120
            for movement, value in max_values.items():
                draw_text_with_background(frame, f"Max {movement.capitalize()}: {value:.2f}", (10, y_offset), 0.7, 2, (255, 255, 255), (0, 0, 0))
                y_offset += 30

        else:
            draw_text_with_background(frame, "Press 'c' to calibrate", (10, 30), 0.7, 2, (0, 255, 0), (0, 0, 0))
    else:
        draw_text_with_background(frame, "No pose detected", (10, 30), 0.7, 2, (0, 0, 255), (0, 0, 0))

    cv2.imshow('Shoulder Range of Motion Measurement', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        print("Starting calibration...")
        neutral_angles = calibration_phase(cap, pose)
        if neutral_angles is not None:
            calibrated = True
            for buffer in angle_buffer.values():
                buffer.clear()
            print(f"Calibrated. Neutral angles set.")
        else:
            print("Calibration failed. Please try again.")
    elif key == ord('r'):
        calibrated = False
        max_values = {k: 0 for k in max_values}
        neutral_angles = None
        for buffer in angle_buffer.values():
            buffer.clear()
        print("Reset. Please recalibrate.")

cap.release()
cv2.destroyAllWindows()