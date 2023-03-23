import pickle
import sys
from datetime import datetime
sys.path.insert(0, '../')
import numpy as np
from keras.models import load_model
import random
import multiprocessing


import processing.preprocessing.face as face_preprocessing 
import processing.preprocessing.eeg as eeg_preprocessing
import processing.preprocessing.gsr as gsr_preprocessing
import processing.preprocessing.ppg as ppg_preprocessing
import processing.feature_extraction.eeg as eeg_feature_extractor
import processing.feature_extraction.gsr as gsr_feature_extractor
import processing.feature_extraction.ppg as ppg_feature_extractor


def ml_prediction(features, data_type):
    #std, mean = pickle.load("models/{0}/scaling_factors.pickle".format(data_type))
    #features = (features - mean) / std

    arousal_scaler = pickle.load(open("models/{0}/arousal_scaler.pickle".format(data_type), 'rb'))
    scaled_features = arousal_scaler.transform([features])
    arousal_model = pickle.load(open("models/{0}/arousal_model.pickle".format(data_type), 'rb'))
    arousal = arousal_model.predict(scaled_features)

    valence_scaler = pickle.load(open("models/{0}/valence_scaler.pickle".format(data_type), 'rb'))
    scaled_features = valence_scaler.transform([features])
    valence_model = pickle.load(open("models/{0}/valence_model.pickle".format(data_type), 'rb'))
    valence = valence_model.predict(scaled_features)

    return {"arousal": int(arousal[0]),
            "valence": int(valence[0]),
            "emotion": None}

def predict_eeg(data, sampling_rate=128, channels=None, method="feature_based"):
    preprocessing = \
        eeg_preprocessing.EegPreprocessing(np.array(data),
                                           channel_names=channels,
                                           sampling_rate=sampling_rate)
    preprocessing.filter_data()
    preprocessing.rereferencing(referencing_value='average')
    # If we know bad channels, we should call this method and pass the list of bad channels, otherwise
    # should not call this method
    #preprocessing.interpolate_bad_channels(bad_channels=["Fp1"]) 
    preprocessed_data = preprocessing.get_data()

    if method == "lstm":
        return None
    else:
        feature_extractor = eeg_feature_extractor.EegFeatures(preprocessed_data, sampling_rate)
        features = feature_extractor.get_total_power_bands()

        return ml_prediction(features, "eeg")

def predict_ppg(data, sampling_rate=128, method = "feature_based"):   
    # data type is list
    data = np.array(data)
    if len(data) < sampling_rate*20:
        print(len(data))
        return {"arousal": 0,
                "valence": 0,
                "emotion": None}
    # data type is list
    data = np.array(data)
    #display_signal(data)
    preprocessing = \
        ppg_preprocessing.PpgPreprocessing(data,
                                           sampling_rate=sampling_rate)
    # neurokit method
    preprocessing.neurokit_filtering()
    preprocessing.filtering()

    # If we don't want to do baseline normalization and just remove baseline should pass False to normalization parameter
    #preprocessing.baseline_normalization(baseline_duration=3, normalization=True)
    preprocessed_data = preprocessing.get_data()  

    if method == "lstm":
        return None
    else:
        features = ppg_feature_extractor.get_feature_vector(preprocessed_data, sampling_rate)
        return ml_prediction(features, "ppg")

def predict_gsr(data, sampling_rate=128, method = "feature_based"):
    if len(data) < sampling_rate*20:
        print(len(data))
        return {"arousal": 0,
                "valence": 0,
                "emotion": None}
    # data type is list
    data = np.array(data)   
    #display_signal(data)
    preprocessing = \
        gsr_preprocessing.GsrPreprocessing(data,
                                           sampling_rate=sampling_rate)
    # neurokit method
    preprocessing.gsr_noise_cancelation()
    # If we don't want to do baseline normalization and just remove baseline should pass False to normalization parameter
    #preprocessing.baseline_normalization(baseline_duration=3, normalization=False)
    preprocessed_data = preprocessing.get_data()
 
    if method == "lstm":
        return None
    else:
        features = gsr_feature_extractor.GsrFeatureExtraction(preprocessed_data, sampling_rate).get_feature_vector()
        return ml_prediction(features, "gsr")

'''
EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
facial_expression_model = load_model("models/face/model-1570411054.h5")
def predict_face(data, frame_rate=30):
    majority_emotion_list = [0, 0, 0, 0, 0, 0, 0]
    faces = face_preprocessing.yolo_face_detection(data, 48, 48)
    for face in faces:
        test_set = np.array([face])
        predicted_values = facial_expression_model.predict(test_set/255, verbose=1)
        label = np.argmax(predicted_values, axis=1)
        print(predicted_values)
        print("label", label, EMOTIONS[label[0]])
        majority_emotion_list[label[0]] += 1
    majority_emotion = np.argmax(np.array(majority_emotion_list))
    if len(data) != 0:
        print("emotion", EMOTIONS[majority_emotion])
    emotion = EMOTIONS[majority_emotion]
    if emotion == "neutral":
        return "neutral"
    elif emotion == "happiness":
        return random.choice(["hahv"])
    elif emotion == "sadness":
        return "lalv"
    elif emotion == "surprise":
        return "hahv"
    elif emotion == "anger":
        return "halv"
    elif emotion == "fear":
        return "halv"
    elif emotion == "disgust":
        return "halv"
    
    return ""
'''

EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

def load_face_model():
    return load_model("models/face/model-1570411054.h5")
   
def predict_face(data, frame_rate=30, model=None):
    print("start predict face")

    facial_expression_model = model
    if not facial_expression_model:
        raise RuntimeError('predict_face: The model is not valid!')
        
    majority_emotion_list = [0, 0, 0, 0, 0, 0, 0]
    faces = face_preprocessing.yolo_face_detection(data, 48, 48)
    for face in faces:
        test_set = np.array([face])
        predicted_values = facial_expression_model.predict(test_set/255, verbose=1)
        label = np.argmax(predicted_values, axis=1)
        majority_emotion_list[label[0]] += 1

    majority_emotion = np.argmax(np.array(majority_emotion_list))
    if len(data) != 0:
        print("emotion", EMOTIONS[majority_emotion])
    emotion = EMOTIONS[majority_emotion]

    if emotion == "neutral":
        return {"arousal": 0,
                "valence": 0,
                "emotion": "Neutral"}
    elif emotion == "happiness":
        return {"arousal": 1,
                "valence": 1,
                "emotion": "Happy"}
    elif emotion == "sadness":
        return {"arousal": -1,
                "valence": -1,
                "emotion": "Sad"}
    elif emotion == "surprise":
        return {"arousal": 1,
                "valence": 1,
                "emotion": "Surprise"}
    elif emotion == "anger":
        return {"arousal": 1,
                "valence": -1,
                "emotion": "Anger"}
    elif emotion == "fear":
        return {"arousal": 1,
                "valence": -1,
                "emotion": "Fear"}
    elif emotion == "disgust":
        return {"arousal": 1,
                "valence": -1,
                "emotion": "disgust"}
    
    return {"arousal": 0,
            "valence": 0,
            "emotion": "Neutral"}


class PredictProcess(multiprocessing.Process):
    def __init__(self, func, in_queue, out_queue, model_loader_func=None):
        super().__init__()
        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._model_loader_func = model_loader_func
    
    def run(self):
        model = None
        if self._model_loader_func:
            model = self._model_loader_func()
            
        while True:
            (args, kwargs) = self.in_queue.get()
            if model:
                kwargs['model'] = model

            result = self.func(*args, **kwargs)
            self.out_queue.put(result)


eeg_predict_process = PredictProcess(predict_eeg, multiprocessing.Queue(), multiprocessing.Queue())
eeg_predict_process.start()
ppg_predict_process = PredictProcess(predict_ppg, multiprocessing.Queue(), multiprocessing.Queue())
ppg_predict_process.start()
gsr_predict_process = PredictProcess(predict_gsr, multiprocessing.Queue(), multiprocessing.Queue())
gsr_predict_process.start()
face_predict_process = PredictProcess(predict_face, multiprocessing.Queue(), multiprocessing.Queue(), load_face_model)
face_predict_process.start()

def predict(data):
    print(data.keys)

    is_eeg_available = False
    is_ppg_available = False
    is_gsr_available = False
    is_camera_available = False

    if "eeg" in data: 
        eeg = data["eeg"]["data"]
        if len(eeg) > 0:
            eeg_sampling_rate = data["eeg"]["sampling_rate"]
            eeg_channels = data["eeg"]["channels"]
            eeg_predict_process.in_queue.put(((eeg,), {"sampling_rate":eeg_sampling_rate,
                                                      "channels":eeg_channels}))
            is_eeg_available = True
            
    if "ppg" in data:
        ppg = data["ppg"]["data"]
        if len(ppg) > 0:
            ppg_sampling_rate = data["ppg"]["sampling_rate"]
            ppg_predict_process.in_queue.put(((ppg,), {"sampling_rate":ppg_sampling_rate})) 
            is_ppg_available = True
    
    if "gsr" in data:
        if len(gsr) > 0:
            gsr = data["gsr"]["data"]
            gsr_sampling_rate = data["gsr"]["sampling_rate"]
            gsr_predict_process.in_queue.put(((gsr,), {"sampling_rate":gsr_sampling_rate}))
            is_gsr_available = True
    
    if "camera" in data:
        camera = data["camera"]["data"]
        if len(camera) > 0:
            camera_frame_rate = data["camera"]["frame_rate"]
            print(datetime.now(), "main process: putting in camera queue: len(data)=", len(camera))
            face_predict_process.in_queue.put(((camera,), {"frame_rate":camera_frame_rate}))
            is_camera_available = True
    
    eeg_prediction = None
    ppg_prediction = None
    gsr_prediction = None
    camera_prediction = None
    eeg = None
    gsr = None
    ppg = None
    face = None

    if is_eeg_available is True:
        try:
            eeg_prediction = eeg_predict_process.out_queue.get()

        except Exception as error:
            eeg_prediction = {"arousal": 0,
                              "valence": 0,
                              "emotion": "Neutral"}
            print("eeg prediction error: ", error)
        eeg = {"prediction": eeg_prediction,
               "arousal_weight": 1,
               "valence_weight": 1}


    if is_gsr_available is True:
        try:
            gsr_prediction = gsr_predict_process.out_queue.get()

        except Exception as error:
            gsr_prediction = {"arousal": 0,
                              "valence": 0,
                              "emotion": "Neutral"}
            print("gsr prediction error: ", error)
        gsr = {"prediction": gsr_prediction,
               "arousal_weight": 2,
               "valence_weight": 1}

    if is_ppg_available is True:
        try:
            ppg_prediction = ppg_predict_process.out_queue.get()

        except Exception as error:
            ppg_prediction = {"arousal": 0,
                              "valence": 0,
                              "emotion": "Neutral"}
            print("ppg prediction error: ", error)
        ppg = {"prediction": ppg_prediction,
               "arousal_weight": 2,
               "valence_weight": 1}
    if is_camera_available is True:
        try:
            print(datetime.now(), "main process: waiting for camera result")
            camera_prediction = face_predict_process.out_queue.get()
            print(datetime.now(), "main process: camera result=", camera_prediction)

        except Exception as error:
            camera_prediction = {"arousal": 0,
                                 "valence": 0,
                                 "emotion": "Neutral"}
            print("camera prediction error: ", error)
        face = {"prediction": camera_prediction,
               "arousal_weight": 1,
               "valence_weight": 2}

    fusion_prediction = decision_fusion(eeg=eeg,
                                        gsr=gsr,   
                                        ppg=ppg,
                                        face=face)
    
    prediction_result = {"eeg": eeg_prediction,
                         "ppg": ppg_prediction,
                         "gsr": gsr_prediction,
                         "camera": camera_prediction,
                         "fusion": fusion_prediction}
    
    return prediction_result

def decision_fusion(eeg=None, ppg=None, gsr=None, face=None):
    arousal_score = 0
    valence_score = 0
    total_arousal = 0
    total_valence = 0
    emotion = None
    if eeg is not None:
        arousal_score += eeg["prediction"]["arousal"] * eeg["arousal_weight"]
        total_arousal += eeg["arousal_weight"]
        valence_score += eeg["prediction"]["valence"] * eeg["valence_weight"]
        total_valence += eeg["valence_weight"]
    if ppg is not None:
        arousal_score += ppg["prediction"]["arousal"] * ppg["arousal_weight"]
        total_arousal += ppg["arousal_weight"]
        valence_score += ppg["prediction"]["valence"] * ppg["valence_weight"]
        total_valence += ppg["valence_weight"]
    if gsr is not None:
        arousal_score += gsr["prediction"]["arousal"] * gsr["arousal_weight"]
        total_arousal += gsr["arousal_weight"]
        valence_score += gsr["prediction"]["valence"] * gsr["valence_weight"]
        total_valence += gsr["valence_weight"]
    if face is not None:
        arousal_score += face["prediction"]["arousal"] * face["arousal_weight"]
        total_arousal += face["arousal_weight"]
        valence_score += face["prediction"]["valence"] * face["valence_weight"]
        total_valence += face["valence_weight"]
        emotion = face["prediction"]["emotion"]  

    arousal = 0
    valence = 0
    if arousal_score < 0:
        arousal = -1
    elif arousal_score > 0:
        arousal = 1
    if valence_score > 0:
        valence = 1
    elif valence_score < 0:
        valence = -1

    print(arousal, valence, emotion)
    return {"arousal": arousal,
            "valence": valence,
            "emotion": emotion}

