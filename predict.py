import pickle
from constants import *
from helpers import get_features, get_most_common_in_list
import sounddevice as sd
import time
from multiprocessing import Queue

model_folder = "models/homepiano"

feature_types = ["frequency", "frequency", "frequency"]
model_types = ["svm", "lda", "knn10"]
#model_types = ["svm", "lda"]

models = []
for feature_type, model_type in zip(feature_types, model_types):
    with open(f"{model_folder}/{feature_type}/{model_type}.pkl", 'rb') as f:
        models.append(pickle.load(f))

previous_chunk_predictions = []
previous_prediction = "silence"
num_stored_preds = 5
for i in range(num_stored_preds):
    previous_chunk_predictions.append("silence")

frame_buffer = Queue(maxsize=3)

def add_to_queue(indata, frames, time, status):
    if not frame_buffer.full():
        frame_buffer.put(indata.T[0])
    else:
        print("Frame dropped")


stream = sd.InputStream(samplerate=fs, channels=1, blocksize=chunk_samples-1, callback=add_to_queue)
stream.start()

cnt = 0
while True:
    before_get_frame = int(time.time() * 1000)
    frame = frame_buffer.get()
    after_get_frame = int(time.time() * 1000)
    #print(f"Time to get frame: {after_get_frame - before_get_frame}")
    model_predictions = []
    for model, feature_type in zip(models, feature_types):
        features = get_features(frame, feature_type)
        model_predictions.append(model.predict([features])[0])

    previous_chunk_predictions[cnt] = get_most_common_in_list(model_predictions)

    prediction = get_most_common_in_list(previous_chunk_predictions)
    if not previous_prediction == prediction:
        print(prediction)
    previous_prediction = prediction

    cnt += 1
    if cnt >= num_stored_preds:
        cnt = cnt % num_stored_preds

    after_predict = int(time.time() * 1000)
    #print(f"Time to predict: {after_predict - after_get_frame}")
    #print(f"Total time per frame: {after_predict - before_get_frame}")