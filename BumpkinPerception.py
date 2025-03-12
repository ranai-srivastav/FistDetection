class AbstractPerceptionModel:
    def __init__(self):
        pass

    def predict(self, frame):
        pass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Uses Google's MediaPipe Hand Landmarker model to detect hand landmarks
class MediapipeWrapper(AbstractPerceptionModel):
    def __init__(self, NUM_MAX_HANDS = 2):
        if not os.path.exists('hand_landmarker.task'):
            print("The model file 'hand_landmarker.task' does not exist. Attempting to fetch...")
            os.system('wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=NUM_MAX_HANDS)
        self.detector = vision.HandLandmarker.create_from_options(options)

    # Helper function to get centroid of hand landmarks
    def _get_hand_centroid(self, hand_landmarks):
        x_avg, y_avg = 0, 0
        for landmark in hand_landmarks:
            x_avg += landmark.x
            y_avg += landmark.y
        x_avg /= len(hand_landmarks)
        y_avg /= len(hand_landmarks)
        return (x_avg, y_avg)

    # Predicts centroids (by percentage of width/height) of hands in the frame
    def predict(self, frame):
        det_results_list = self.detector.detect(frame)
        det_centroids = []
        for i, det_results in enumerate(det_results_list.hand_landmarks):
            det_centroids.append(self._get_hand_centroid(det_results))
        return det_centroids

if __name__ == "__main__":
    img = mp.Image.create_from_file('image.jpg')
    WIDTH, HEIGHT = 640, 960
    model = MediapipeWrapper()
    centroids = model.predict(img)
    for centroid in centroids:
        x, y = int(centroid[0] * WIDTH), int(centroid[1] * HEIGHT)
        print("Hand centroid at: ({}, {})".format(x, y))