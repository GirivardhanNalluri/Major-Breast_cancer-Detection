from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import pickle
import cv2
class BreastCancerClassifier:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.model = None
        self.svm = None
        self.scaler = None
    def load_models(self):
        self.model = tf.keras.models.load_model('Models/stage_classification_model.keras')
        with open('Models/stages_svm_model.pkl', 'rb') as f:
            self.svm, self.scaler = pickle.load(f)
    def predict_stage(self, image_path):
        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        hybrid_pred = self.model.predict(img_array)
        hybrid_stage = np.argmax(hybrid_pred) + 1
        feature_model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        features = feature_model.predict(img_array)
        features_scaled = self.scaler.transform(features)
        svm_stage = self.svm.predict(features_scaled)[0] + 1
        final_stage = round((hybrid_stage + svm_stage) / 2)
        return final_stage, hybrid_pred[0][final_stage-1]
class BreastCancerDetector:
    def __init__(self, benign_malignant_model_path, stage_model_path, svm_model_path):
        self.resnet_model = load_model(benign_malignant_model_path)
        self.stage_classifier = BreastCancerClassifier()
        self.stage_classifier.load_models()
    def process_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    def predict(self, image_path):
        img_array = self.process_image(image_path)
        pred_prob = self.resnet_model.predict(img_array)
        is_malignant = pred_prob[0][0] < 0.6
        result = {
            'diagnosis': 'Malignant' if is_malignant else 'Benign',
            'confidence': float(abs(1 - pred_prob[0][0]))
        }
        if is_malignant:
            stage, stage_confidence = self.stage_classifier.predict_stage(image_path)
            result['stage'] = int(stage)
            result['stage_confidence'] = float(stage_confidence)
        return result

def test_image(image_path):
    detector = BreastCancerDetector(
        benign_malignant_model_path='Models/ResNet50_model.h5',
        stage_model_path='Models/stage_classification_model.keras',
        svm_model_path='Models/stages_svm_model.pkl'
    )

    result = detector.predict(image_path)

    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")

    if result['diagnosis'] == 'Malignant':
        print(f"Stage: {result['stage']}")
        print(f"Stage Confidence: {result['stage_confidence']:.2%}")

    return result
