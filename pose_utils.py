import cv2


class _EmptyResult:
    pose_landmarks = None


class PoseDetector:
    def detect_pose(self, frame):
        return _EmptyResult()

    def draw_landmarks(self, frame, result):
        return frame
