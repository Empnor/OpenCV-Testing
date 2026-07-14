import cv2

DETECTOR_PATH = "models/face_detection_yunet_2023mar_int8.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"

_detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH, "", (320, 320),
    score_threshold=0.7, nms_threshold=0.3, top_k=5000
)
_recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")


def get_embedding(image):
    "Detect the largest face in `image` and return its SFace embedding, or None."
    h, w = image.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    face = faces[0]

    aligned = _recognizer.alignCrop(image, face)
    embedding = _recognizer.feature(aligned)
    return embedding


def get_recognizer():
    "Access to the shared SFace recognizer, e.g. for .match() calls."
    return _recognizer