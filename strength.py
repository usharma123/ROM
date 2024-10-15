import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.KEY_LANDMARKS = [
            self.mp_hands.HandLandmark.WRIST,
            self.mp_hands.HandLandmark.THUMB_CMC,
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_MCP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

    def find_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.hands.process(image_rgb)

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return image
    
    def get_key_landmarks(self, hand_landmarks):
        return {landmark: np.array([hand_landmarks.landmark[landmark].x,
                                    hand_landmarks.landmark[landmark].y,
                                    hand_landmarks.landmark[landmark].z])
                for landmark in self.KEY_LANDMARKS}

class KinematicAnalysis:
    def __init__(self, buffer_size=30, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.position_buffers = {landmark: deque(maxlen=buffer_size) for landmark in HandTracker().KEY_LANDMARKS}
        
    def update(self, landmarks):
        for landmark, position in landmarks.items():
            self.position_buffers[landmark].append(position)
        
    def calculate_velocity(self, landmark):
        buffer = self.position_buffers[landmark]
        if len(buffer) < 2:
            return np.zeros(3)
        return (buffer[-1] - buffer[-2]) * self.fps
    
    def calculate_acceleration(self, landmark):
        buffer = self.position_buffers[landmark]
        if len(buffer) < 3:
            return np.zeros(3)
        v1 = (buffer[-1] - buffer[-2]) * self.fps
        v2 = (buffer[-2] - buffer[-3]) * self.fps
        return (v1 - v2) * self.fps
    
    def calculate_angular_velocity(self, landmark1, landmark2):
        if len(self.position_buffers[landmark1]) < 2 or len(self.position_buffers[landmark2]) < 2:
            return 0
        vec1 = self.position_buffers[landmark1][-1] - self.position_buffers[landmark2][-1]
        vec2 = self.position_buffers[landmark1][-2] - self.position_buffers[landmark2][-2]
        angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
        return angle * self.fps

class HandCalibration:
    def __init__(self, tracker, num_frames=60):
        self.tracker = tracker
        self.num_frames = num_frames
        self.hand_lengths = []
        self.pixel_to_meter_ratios = []

    def calibrate(self, cap):
        print("Calibration starting. Please hold your hand still, palm facing the camera.")
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.tracker.find_hands(frame)
            frame = self.tracker.draw_landmarks(frame, results)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                key_landmarks = self.tracker.get_key_landmarks(hand_landmarks)
                
                wrist = key_landmarks[self.tracker.mp_hands.HandLandmark.WRIST]
                middle_tip = key_landmarks[self.tracker.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                hand_length = np.linalg.norm(middle_tip - wrist)
                self.hand_lengths.append(hand_length)

                pixel_length = np.linalg.norm(middle_tip[:2] - wrist[:2]) * frame.shape[1]
                self.pixel_to_meter_ratios.append(0.189 / pixel_length)

            cv2.putText(frame, f"Calibrating: {_ + 1}/{self.num_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Calibration")

        avg_hand_length = np.mean(self.hand_lengths)
        avg_pixel_to_meter = np.mean(self.pixel_to_meter_ratios)
        hand_mass = 70 * 0.006

        print(f"Calibration complete. Estimated hand length: {avg_hand_length:.4f} m, Hand mass: {hand_mass:.4f} kg")
        return hand_mass, avg_hand_length, avg_pixel_to_meter

class InverseDynamics:
    def __init__(self, mass, length, com_ratio):
        self.mass = mass
        self.length = length
        self.com_ratio = com_ratio
        self.g = 9.81
        
    def calculate_dynamics(self, angular_velocity, linear_acceleration, external_force=(0,0,0)):
        com = self.length * self.com_ratio
        inertia = (1/12) * self.mass * self.length**2
        
        weight = np.array([0, 0, self.mass * self.g])
        inertial_force = self.mass * linear_acceleration
        
        moment_of_weight = np.cross(com * np.array([1, 0, 0]), weight)
        moment_of_inertia = inertia * angular_velocity
        moment_of_external_force = np.cross(self.length * np.array([1, 0, 0]), external_force)
        
        joint_force = weight + inertial_force - external_force
        joint_moment = moment_of_weight + moment_of_inertia - moment_of_external_force
        
        return joint_force, joint_moment

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    hand_tracker = HandTracker()
    calibration = HandCalibration(hand_tracker)
    
    print("Place your hand in a neutral position and press 'c' to calibrate")
    
    calibrated = False
    while not calibrated:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, "Press 'c' to calibrate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            hand_mass, hand_length, pixel_to_meter = calibration.calibrate(cap)
            calibrated = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    kinematics = KinematicAnalysis(buffer_size=30, fps=30)
    dynamics = InverseDynamics(mass=hand_mass, length=hand_length, com_ratio=0.5)
    
    force_data = []
    moment_data = []
    
    print("Calibration complete. Starting analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = hand_tracker.find_hands(frame)
        frame = hand_tracker.draw_landmarks(frame, results)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            key_landmarks = hand_tracker.get_key_landmarks(hand_landmarks)
            
            for landmark in key_landmarks:
                key_landmarks[landmark] *= pixel_to_meter
            
            kinematics.update(key_landmarks)
            
            wrist_acceleration = kinematics.calculate_acceleration(hand_tracker.mp_hands.HandLandmark.WRIST)
            angular_velocity = kinematics.calculate_angular_velocity(
                hand_tracker.mp_hands.HandLandmark.WRIST,
                hand_tracker.mp_hands.HandLandmark.INDEX_FINGER_TIP
            )
            
            force, moment = dynamics.calculate_dynamics(angular_velocity, wrist_acceleration)
            force_mag = np.linalg.norm(force)
            moment_mag = np.linalg.norm(moment)
            
            force_data.append(force_mag)
            moment_data.append(moment_mag)
            
            cv2.putText(frame, f"Force: {force_mag:.2f} N", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Moment: {moment_mag:.2f} Nm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for landmark in key_landmarks.values():
                cv2.circle(frame, (int(landmark[0] / pixel_to_meter), int(landmark[1] / pixel_to_meter)), 
                           5, (255, 0, 0), -1)
        
        cv2.imshow("Hand Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(force_data)
    plt.title("Joint Force Magnitude")
    plt.ylabel("Force (N)")
    
    plt.subplot(2, 1, 2)
    plt.plot(moment_data)
    plt.title("Joint Moment Magnitude")
    plt.ylabel("Moment (Nm)")
    plt.xlabel("Frame")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()