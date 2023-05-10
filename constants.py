fs = 44100  # Sample rate
seconds = 2  # Duration of recording for training
chunk_length = 50 # length of chunks for training, predicting (milliseconds)
chunk_length_seconds = chunk_length / 1000
chunk_samples = int(fs * chunk_length_seconds)

feature_types = ["frequency", "chroma"]
model_types = ["bayes", "lda", "svm", "tree"]

notes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
chord_types = ["major", "minor"]