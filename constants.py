fs = 44100  # Sample rate
seconds = 2  # Duration of recording for training
chunk_length = 50 # length of chunks for training, predicting (milliseconds)
chunk_length_seconds = chunk_length / 1000
chunk_samples = int(fs * chunk_length_seconds)

feature_types = ["pca", "frequency", "pcm"]
model_types = ["bayes", "knn2", "knn5", "knn10", "knn20", "lda", "svm", "tree"]

octaves = ["3", "4"]
notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
chord_types = ["major", "minor"]

IP = "127.0.0.1"
PORT = 6780