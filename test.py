import argparse
import os
from scipy.io.wavfile import read
import pickle
from constants import *
from helpers import present_options, get_features
from data_vis_methods import generate_missclassification_graph, make_accuracy_heatmap, tally_misclassification_types, make_misclassification_bar_graph
import numpy as np
import time


######################################    Parse arguments    ######################################

parser = argparse.ArgumentParser()
parser.add_argument("--testset_folder")
parser.add_argument("--model_folder")
parser.add_argument("--select_models", required=False)
args = parser.parse_args()

testset_folder = args.testset_folder
if (testset_folder == None):
    print("Please enter the testset folder")
    exit(1)
if testset_folder[-1] == "/":
    testset_folder = testset_folder[:-1]
if not os.path.isdir(testset_folder):
    os.makedirs(testset_folder)
testset_folder_name = testset_folder.split('/')[-1]

model_folder = args.model_folder
if (model_folder == None):
    print("Please enter the model folder")
    exit(1)
if model_folder[-1] == "/":
    model_folder = model_folder[:-1]
if not os.path.isdir(model_folder):
    print("Model folder does not exist")
    exit(1)
model_folder_name = model_folder.split('/')[-1]

if args.select_models == "true":
    feature_types = present_options(feature_types)
    model_types = present_options(model_types)

plots_folder = f"plots/{model_folder_name}/{testset_folder_name}"
if not os.path.isdir(plots_folder):
    os.makedirs(plots_folder)
results_file = open(f'{plots_folder}/accuracies.txt', 'w')


######################################    Load test set    ######################################
test_files = []
for address, dirs, file_names in os.walk(testset_folder):
    for name in file_names:
        test_files.append(os.path.join(address, name))

vectors = []
actual_labels = []
for f in test_files:
    fs, data = read(f)
    label = f.split('/')[-1].split('_')[0]

    for chunk_num in range(int(len(data) / chunk_samples)):
        data_chunk = data[chunk_num * chunk_samples : (chunk_num + 1) * chunk_samples - 1]
        vectors.append(data_chunk)
        actual_labels.append(label)

######################################    Predict and calculate accuracies of models    ######################################

accuracies = np.zeros((len(model_types), len(feature_types)))
related_vals = np.zeros((len(model_types), len(feature_types)))
unrelated_vals = np.zeros((len(model_types), len(feature_types)))
silence_vals = np.zeros((len(model_types), len(feature_types)))

for feature_idx, feature_type in enumerate(feature_types):
    print(f"=================  Feature type:  {feature_type}  ================")
    results_file.write(f"=================  Feature type:  {feature_type}  ================\n")
    for model_idx, model_type in enumerate(model_types):
        start_time = time.time()
        with open(f"{model_folder}/{feature_type}/{model_type}.pkl", 'rb') as f:
            clf = pickle.load(f)

        predicted_labels = []
        for v in vectors:
            features = get_features(v, feature_type)
            predicted_labels.append(clf.predict([features])[0])

        correct_preds = 0
        for (actual, predicted) in zip(actual_labels, predicted_labels):
            if actual == str(predicted):
                correct_preds += 1

        accuracy = round(correct_preds * 100 / len(actual_labels))
        accuracies[model_idx, feature_idx] = accuracy

        time_secs = int(time.time() - start_time)
        
        print(f"{model_type}     {accuracy}%    time: {time_secs} seconds")
        results_file.write(f"{model_type}     {accuracy}%    time: {time_secs} seconds\n")

        generate_missclassification_graph(actual_labels, predicted_labels, feature_type, model_type, accuracy, plots_folder)
        related_vals[model_idx, feature_idx], unrelated_vals[model_idx, feature_idx], silence_vals[model_idx, feature_idx] = tally_misclassification_types(actual_labels, predicted_labels)

make_accuracy_heatmap(accuracies, plots_folder)
make_misclassification_bar_graph(model_types, feature_types, related_vals, unrelated_vals, silence_vals, plots_folder)
results_file.close()