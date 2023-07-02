import json
import numpy as np
import mediapipe as mp
import cv2


class AttentionTracker:
    def __init__(self, config_path: str, emotions_detector) -> None:
        # read config face points
        with open(config_path) as json_config:
            dict_config = json.load(json_config)

        self.top_left_eye_data = dict_config["top_left_eye"]
        self.bottom_left_eye_data = dict_config["bottom_left_eye"]
        self.top_right_eye_data = dict_config["top_right_eye"]
        self.bottom_right_eye_data = dict_config["bottom_right_eye"]
        self.left_iris_data = dict_config["left_iris"]
        self.right_iris_data = dict_config["right_iris"]
        self.left_face_border_data = dict_config["left_face_border"]
        self.right_face_border_data = dict_config["right_face_border"]

        right_indices = np.array(self.top_right_eye_data + self.bottom_right_eye_data)
        left_indices = np.array(self.top_left_eye_data + self.bottom_left_eye_data)
        self.right_eyeblink_landmarks = right_indices[[0, 8, 12, 4]]
        self.left_eyeblink_landmarks = left_indices[[0, 8, 12, 4]]

        self.head_orientation_landmarks = [33, 263, 1, 61, 61, 291, 199]

        self.emotions_detector = emotions_detector
        # self.face_mesh = face_mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_emotions(self, frame):
        return self.emotions_detector.detect_emotions(frame)

    def euclaidean_distance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    def blink_ratio(self, right_coords, left_coords):
        rhDistance = self.euclaidean_distance(right_coords[0], right_coords[1])
        rvDistance = self.euclaidean_distance(right_coords[2], right_coords[3])

        lvDistance = self.euclaidean_distance(left_coords[0], left_coords[1])
        lhDistance = self.euclaidean_distance(left_coords[2], left_coords[3])

        reRatio = rhDistance / rvDistance
        leRatio = lhDistance / lvDistance

        ratio = (reRatio + leRatio) / 2
        return ratio

    def get_head_orientation(self, landmarks, img_w, img_h):
        face_3d = np.array(
            [
                np.concatenate(
                    (
                        self.get_landmark_coord(
                            face_landmark=lm, img_w=img_w, img_h=img_h
                        ),
                        np.array([lm.z]),
                    ),
                    axis=0,
                )
                for lm in landmarks[self.head_orientation_landmarks]
            ]
        ).astype(np.float64)
        face_2d = face_3d[:, [0, 1]].astype(np.float64)

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array(
            [
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1],
            ]
        )

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles = cv2.RQDecomp3x3(rmat)[0]

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        # z = angles[2] * 360

        # See where the user's head tilting
        if y < -10:
            head_orientation = "Looking Left"
        elif y > 10:
            head_orientation = "Looking Right"
        elif x < -10:
            head_orientation = "Looking Down"
        elif x > 10:
            head_orientation = "Looking Up"
        else:
            head_orientation = "Forward"

        return head_orientation

    def get_landmark_coord(self, face_landmark, img_w, img_h):
        return np.multiply([face_landmark.x, face_landmark.y], [img_w, img_h]).astype(
            int
        )

    def is_eyes_closed(self, landmarks, img_w, img_h):
        right_eyelid_coords = np.array(
            [
                self.get_landmark_coord(face_landmark=lm, img_w=img_w, img_h=img_h)
                for lm in landmarks[self.right_eyeblink_landmarks]
            ]
        )
        left_eyelid_coords = np.array(
            [
                self.get_landmark_coord(face_landmark=lm, img_w=img_w, img_h=img_h)
                for lm in landmarks[self.left_eyeblink_landmarks]
            ]
        )
        ratio = self.blink_ratio(
            right_coords=right_eyelid_coords, left_coords=left_eyelid_coords
        )

        # if ratio > 5.3:
        #     return True
        # return False
        return ratio

    def get_emotion(self, emotion: str, detected_emotions: dict) -> float:
        if len(detected_emotions) > 0:
            return detected_emotions[0]["emotions"][emotion]
        else:
            return None

    def process_face_mesh(self, image):
        image.flags.writeable = False
        face_mesh_processed = self.face_mesh.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape

        if face_mesh_processed.multi_face_landmarks:
            lms = np.array(face_mesh_processed.multi_face_landmarks[0].landmark)

            return {
                "eyes_closed": self.is_eyes_closed(
                    landmarks=lms, img_h=img_h, img_w=img_w
                ),
                "head_turn": self.get_head_orientation(
                    landmarks=lms, img_h=img_h, img_w=img_w
                ),
            }
        return {}

    def process_frame(self, frame) -> dict:
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        detected_emotions = self.detect_emotions(image)

        return {
            "is_face_detected": len(detected_emotions) > 0,
            "emotions": {
                "angry": self.get_emotion(
                    emotion="angry", detected_emotions=detected_emotions
                ),
                "disgust": self.get_emotion(
                    emotion="disgust", detected_emotions=detected_emotions
                ),
                "fear": self.get_emotion(
                    emotion="fear", detected_emotions=detected_emotions
                ),
                "happy": self.get_emotion(
                    emotion="happy", detected_emotions=detected_emotions
                ),
                "sad": self.get_emotion(
                    emotion="sad", detected_emotions=detected_emotions
                ),
                # "surprize": self.get_emotion(
                #     emotion="surprize", detected_emotions=detected_emotions
                # ),
                "neutral": self.get_emotion(
                    emotion="neutral", detected_emotions=detected_emotions
                ),
            },
            "head_turn": self.process_face_mesh(image).get("head_turn", None),
            "eyes_closed": self.process_face_mesh(image).get("eyes_closed", None),
        }
