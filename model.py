# This is a minimal version just for processing 1 frame
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        h, w, _ = frame.shape

        def get_coords(landmark):
            return (int(landmarks[landmark].x * w), int(landmarks[landmark].y * h))

        ear = get_coords(mp_pose.PoseLandmark.LEFT_EAR)
        shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
        knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE)
        ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)

        knee_angle = calculate_angle(hip, knee, ankle)
        thigh_torso_angle = calculate_angle(shoulder, hip, knee)
        neck_angle = calculate_angle(ear, shoulder, hip)

        tracked_points = [ear, shoulder, hip, knee, ankle]
        for point in tracked_points:
            cv2.circle(frame, point, 8, (0, 255, 255), -1)

        cv2.putText(frame, f"Knee: {int(knee_angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Thigh-Torso: {int(thigh_torso_angle)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Neck: {int(neck_angle)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if knee_angle < 80:
            cv2.putText(frame, "Knee too bent!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif thigh_torso_angle < 70:
            cv2.putText(frame, "Sit upright!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif neck_angle < 170:
            cv2.putText(frame, "Straighten your neck!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Good Posture!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
