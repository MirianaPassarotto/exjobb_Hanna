import cv2
import mediapipe as mp
import numpy as np
from math import atan2
import pyrealsense2 as rs  # Optional, not used in this webcam version
import requests

class PoseEstimator:
    def __init__(self, server_url):
        self.server_url = server_url
        self.speed = "slow"
        self.last_speed = None

        self.left_active = False
        self.right_active = False

        self.detected_hands = []
        self.active_hands = []
        self.handedness = []

        # Angle limits for gesture detection
        self.lower_limit_armpit = 50
        self.upper_limit_armpit = 130
        self.lower_limit_elbow = 100
        self.upper_limit_elbow = 180

        # Mediapipe modules
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def angle(self, p1, p2, p3):
        a = atan2(p3[1] - p2[1], p3[0] - p2[0]) - atan2(p1[1] - p2[1], p1[0] - p2[0])
        a = np.rad2deg(a)
        return abs(a if a <= 180 else 360 - a)

    def is_open(self, hand_landmarks):
        tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        mcps = [
            self.mp_hands.HandLandmark.THUMB_MCP,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]
        wrist = np.array([
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y,
            hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].z
        ])

        extended = 0
        for tip, mcp in zip(tips, mcps):
            tip_pos = np.array([hand_landmarks.landmark[tip].x, hand_landmarks.landmark[tip].y, hand_landmarks.landmark[tip].z])
            mcp_pos = np.array([hand_landmarks.landmark[mcp].x, hand_landmarks.landmark[mcp].y, hand_landmarks.landmark[mcp].z])
            if np.linalg.norm(tip_pos - wrist) > np.linalg.norm(mcp_pos - wrist) + 0.05:
                extended += 1

        return True if extended == 5 else False if extended == 0 else None

    def get_speed(self):
        for i, hand in enumerate(self.active_hands):
            label = self.handedness[i].classification[0].label
            tips = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            mcps = [
                self.mp_hands.HandLandmark.THUMB_MCP,
                self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                self.mp_hands.HandLandmark.RING_FINGER_MCP,
                self.mp_hands.HandLandmark.PINKY_MCP
            ]
            wrist = np.array([
                hand.landmark[self.mp_hands.HandLandmark.WRIST].x,
                hand.landmark[self.mp_hands.HandLandmark.WRIST].y,
                hand.landmark[self.mp_hands.HandLandmark.WRIST].z
            ])

            ext = []
            for tip, mcp in zip(tips, mcps):
                tip_pos = np.array([hand.landmark[tip].x, hand.landmark[tip].y, hand.landmark[tip].z])
                mcp_pos = np.array([hand.landmark[mcp].x, hand.landmark[mcp].y, hand.landmark[mcp].z])
                ext.append(np.linalg.norm(tip_pos - wrist) > np.linalg.norm(mcp_pos - wrist) + 0.05)

            if ext[0] and not any(ext[1:]):
                return "slow"
            if ext[1] and ext[2] and not any([ext[0], ext[3], ext[4]]):
                return "fast"

        return self.speed

    def get_gesture(self, laa, raa, lea, rea, x_vals, open_hand):
        pointing_left = (
            self.lower_limit_armpit < laa < self.upper_limit_armpit and
            self.lower_limit_elbow < lea < self.upper_limit_elbow and
            not (self.lower_limit_armpit < raa < self.upper_limit_armpit) and
            (x_vals[2] > x_vals[0] + 0.2)
        )
        pointing_right = (
            self.lower_limit_armpit < raa < self.upper_limit_armpit and
            self.lower_limit_elbow < rea < self.upper_limit_elbow and
            not (self.lower_limit_armpit < laa < self.upper_limit_armpit) and
            (x_vals[1] > x_vals[3] + 0.2)
        )

        if pointing_left:
            return "left"
        elif pointing_right:
            return "right"
        elif open_hand is not None:
            return "openhand" if open_hand else "closedhand"
        else:
            return None

    def run(self):
        cap = cv2.VideoCapture(0)

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_results = pose.process(image)
                hands_results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if not pose_results.pose_landmarks:
                    continue

                lms = pose_results.pose_landmarks.landmark
                self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                def get_coords(idx):
                    return np.array([lms[idx].x, lms[idx].y, lms[idx].z])

                ls, rs = get_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                le, re = get_coords(self.mp_pose.PoseLandmark.LEFT_ELBOW.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                lw, rw = get_coords(self.mp_pose.PoseLandmark.LEFT_WRIST.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
                lh, rh = get_coords(self.mp_pose.PoseLandmark.LEFT_HIP.value), get_coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)

                laa, raa = self.angle(lh, ls, le), self.angle(rh, rs, re)
                lea, rea = self.angle(ls, le, lw), self.angle(rs, re, rw)

                self.left_active = abs(lw[1] - ls[1]) < 0.5 and abs(lw[0] - ls[0]) < 0.5
                self.right_active = abs(rw[1] - rs[1]) < 0.5 and abs(rw[0] - rs[0]) < 0.5

                self.detected_hands = []
                self.active_hands = []
                self.handedness = hands_results.multi_handedness if hands_results.multi_handedness else []
                hand_open = None

                if hands_results.multi_hand_landmarks:
                    for i, hl in enumerate(hands_results.multi_hand_landmarks):
                        label = self.handedness[i].classification[0].label
                        is_left = label == "Left"
                        active = (is_left and self.left_active) or (not is_left and self.right_active)

                        if active:
                            wrist = hl.landmark[self.mp_hands.HandLandmark.WRIST]
                            shoulder = ls if is_left else rs
                            near_shoulder = abs(wrist.x - shoulder[0]) < 0.5 and abs(wrist.y - shoulder[1]) < 0.5

                            if near_shoulder:
                                self.mp_drawing.draw_landmarks(image, hl, self.mp_hands.HAND_CONNECTIONS)
                                self.detected_hands.append(hl)
                                self.active_hands.append(hl)

                                open_check = self.is_open(hl)
                                if open_check is True:
                                    hand_open = True
                                elif open_check is False and hand_open is not True:
                                    hand_open = False

                x_vals = [ls[0], rs[0], lw[0], rw[0], le[0], re[0], lh[0], rh[0]]
                gesture = self.get_gesture(laa, raa, lea, rea, x_vals, hand_open)

                if self.active_hands:
                    new_speed = self.get_speed()
                    if new_speed != self.last_speed:
                        print(f"Detected speed: {new_speed}")
                        # requests.post(self.server_url, json={'speed': new_speed})
                        self.speed = new_speed
                        self.last_speed = new_speed

                if gesture:
                    print(f"Detected gesture: {gesture}")
                    # requests.post(self.server_url, json={'command': gesture})

                cv2.imshow('Mediapipe Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    server_url = "http://192.168.0.2:5000/command"
    PoseEstimator(server_url).run()
