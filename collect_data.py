import sounddevice as sd
from scipy.io.wavfile import write
import argparse
import os
import time
from helpers import present_options
from constants import *

######################################    Parse arguments    ######################################

parser = argparse.ArgumentParser()
parser.add_argument("--observation_folder")
parser.add_argument("--num_observations", required=False)
parser.add_argument("--select_classes", required=False)
args = parser.parse_args()

obs_folder = args.observation_folder
if (obs_folder == None):
    print("Please enter the observation folder")
    exit()
if obs_folder[-1] == "/":
    obs_folder = obs_folder[:-1]
if not os.path.exists(obs_folder):
    os.makedirs(obs_folder)

num_observations = args.num_observations
if num_observations == None:
    num_observations = 1
num_obs = int(num_observations)
files = os.listdir(obs_folder)

if args.select_classes == "true":
    notes = present_options(notes)
    chord_types = present_options(chord_types)
    record_silence = input("Do you want to add silent observations?  (\"yes\", \"no\")\n") == "yes"
else:
    record_silence = True



######################################    Record data    ######################################

def record_data(label, message):
    counter = 0
    for f in files:
        if label in f:
            counter += 1
    for i in range(num_obs):
        print(message)
        time.sleep(1)
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        print("Data collected")
        write(f'{obs_folder}/{label}_{counter}.wav', fs, recording)
        time.sleep(1)
        counter += 1

print("========  Collecting Data  ========")
time.sleep(2)
for note in notes:
    for chord_type in chord_types:
        label = f"{note}-{chord_type}"
        message = f"Play {note} {chord_type}"
        record_data(label, message)

if record_silence:
    label = "silence"
    message = "Don't play anything"
    record_data(label, message)