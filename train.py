from scipy.io import wavfile
from sklearn import svm, neighbors, tree, naive_bayes, discriminant_analysis
import os
import argparse
import pickle
from sklearn.decomposition import PCA
from constants import *
from helpers import present_options, normalise_pcm, fourier_transform


######################################    Parse arguments    ######################################

parser = argparse.ArgumentParser()
parser.add_argument("--observation_folder")
parser.add_argument("--model_folder")
parser.add_argument("--select_models", required=False)
args = parser.parse_args()

obs_folder = args.observation_folder
if (obs_folder == None):
    print("Please enter the observation folder")
    exit()
if obs_folder[-1] == "/":
    obs_folder = obs_folder[:-1]
if not os.path.isdir(obs_folder):
    print("Observation folder is not an existing directory")
    exit()

model_folder = args.model_folder
if model_folder == None:
    print("Please enter the model folder")
    exit()
if model_folder[-1] == "/":
    model_folder = model_folder[:-1]
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
for feature_type in feature_types:
    if not os.path.isdir(f"{model_folder}/{feature_type}"):
        os.mkdir(f"{model_folder}/{feature_type}")

if args.select_models == "true":
    feature_types = present_options(feature_types)
    model_types = present_options(model_types)

######################################    Create features    ######################################

print(f"Collecting data from {obs_folder}")
    
pcms = []
frequencies = []
labels = []

files = []
for address, dirs, file_names in os.walk(obs_folder):
    for name in file_names:
        files.append(os.path.join(address, name))

for f in files:
    label = f.split('/')[-1].split('_')[0]

    fs, data = wavfile.read(f)
    for chunk_num in range(int(len(data) / chunk_samples)):
        data_chunk = data[chunk_num * chunk_samples : (chunk_num + 1) * chunk_samples - 1]

        ch1_normalised = normalise_pcm(data_chunk)
        fourier_spectrum = fourier_transform(ch1_normalised)
        
        pcms.append(ch1_normalised)
        frequencies.append(fourier_spectrum)

        labels.append(label)

pca = PCA(n_components = 5)
pca.fit(frequencies)
pca_features = pca.transform(frequencies)
with open(f"{model_folder}/pca/pca_class.pkl", 'wb') as f:
    pickle.dump(pca, f)


######################################    Train models    ######################################

print(f"Training models into {model_folder}:")

model_classes = {
    "bayes": naive_bayes.GaussianNB(),
    "knn2": neighbors.KNeighborsClassifier(2),
    "knn5": neighbors.KNeighborsClassifier(5),
    "knn10": neighbors.KNeighborsClassifier(10),
    "knn20": neighbors.KNeighborsClassifier(20),
    "knn40": neighbors.KNeighborsClassifier(40),
    "lda": discriminant_analysis.LinearDiscriminantAnalysis(),
    "svm": svm.SVC(),
    "tree": tree.DecisionTreeClassifier()
}
features = {
    "frequency": frequencies,
    "pcm": pcms,
    "pca": pca_features,
}

for feature_type in feature_types:
    print(f"=======  Training models with {feature_type} features  =======")
    for model_type in model_types:
        clf = model_classes[model_type]
        clf.fit(features[feature_type], labels)
        with open(f"{model_folder}/{feature_type}/{model_type}.pkl", 'wb') as f:
            pickle.dump(clf, f)
        print(f"Trained {model_type} model")

print(f"TRAINING COMPLETE \U0001F973")
        