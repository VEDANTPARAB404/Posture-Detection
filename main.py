from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

status_text = "Click Start to begin"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle

def gen_frames():
    global status_text
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            h, w, _ = frame.shape

            def get_coords(idx):
                return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

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

            if knee_angle < 80:
                status_text = "Knee too bent!"
                color = (0, 0, 255)
            elif thigh_torso_angle < 70:
                status_text = "Sit upright!"
                color = (0, 0, 255)
            elif neck_angle < 170:
                status_text = "Straighten your neck!"
                color = (0, 0, 255)
            else:
                status_text = "Good Posture!"
                color = (0, 255, 0)

            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return status_text

if __name__ == '__main__':
    app.run(debug=True)
