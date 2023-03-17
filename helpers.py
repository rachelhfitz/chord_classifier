from scipy.fftpack import rfft
import numpy as np
import pickle
from constants import *

def get_all_labels():
    labels = []
    for octave in octaves:
        for note in notes:
            for chord_type in chord_types:
                labels.append(f"{note}-{octave}-{chord_type}")
    labels.append("silence")
    return labels

# prompts the user to select some of the options provided. Returns a list of those selected
def present_options(options):
    requested_items = []
    request = input(f"Please enter your selected options:  {options}\n")
    while request:
        if request in options:
            requested_items.append(request)
        else:
            print(f"{request} is not one of the available options")
        request = input()
    print(requested_items)
    return list(set(requested_items))

# data: raw pcm file
def normalise_pcm(data):
    ch1 = data.T
    max_pwr = max(abs(ch1))
    return [s / max_pwr for s in ch1]

# data: normalised pcm
def fourier_transform(data):
    return np.abs(rfft(data))


def get_features(vector, feature_type, model_folder):
    normalised_vector = normalise_pcm(vector)
    frequency_spectrum = fourier_transform(normalised_vector)

    if feature_type == "frequency":
        return frequency_spectrum
    elif feature_type == "pcm":
        return normalised_vector
    elif feature_type == "pca":
        with open(f"{model_folder}/pca/pca_class.pkl", 'rb') as f:
            pca = pickle.load(f)
        return pca.transform([frequency_spectrum])[0]
    else:
        print(f"I don't know how to calculate {feature_type} features")
        exit()

# given a full label like "G-3-minor", returns "G3m"
# returns "s" for "silence"
def get_short_label(full_label):
    if full_label == "silence":
        return "S"
    note, octave, chord_type = split_label(full_label)
    quality = "M" if chord_type == "major" else "m"
    return f"{note}{octave}{quality}"

def split_label(label):
    components = label.split('-')
    note = components[0]
    octave = components[1]
    chord_type = components[2]
    return note, octave, chord_type

def get_note_num_semitones_above(note, octave, num_semitones):
    oct = int(octave)
    for ind, n in enumerate(notes):
        if n == note:
            break
    for i in range(num_semitones):
        ind += 1
        if ind >= len(notes):
            oct += 1
            ind = ind % len(notes)
            if not str(oct) in octaves: # we've gone out of range
                return (None, None)
    return (notes[ind], str(oct))


# given 2 chord labels, returns true if chords share 2 or more notes
def chords_are_related(l1, l2):
    if l1 == "silence" or l2 == "silence":
        return False
    if l1 == l2:
        return True
    l1_note, l1_octave, l1_chord_type = split_label(l1)
    l2_note, l2_octave, l2_chord_type = split_label(l2)
    if l1_note == l2_note and l1_octave == l2_octave:
        return True  # these are parallel major/minor chords
    if l1_chord_type == "major":
        if (l2_note, l2_octave) == get_note_num_semitones_above(l1_note, l1_octave, 4) or (l1_note, l1_octave) == get_note_num_semitones_above(l2_note, l2_octave, 3):
            # l2 is a major 3rd above l1 or l1 is a minor 3rd above l2
            return True
    elif l1_chord_type == "minor":
        if (l1_note, l1_octave) == get_note_num_semitones_above(l2_note, l2_octave, 4) or (l2_note, l2_octave) == get_note_num_semitones_above(l1_note, l1_octave, 3):
            # l1 is a major 3rd above l2 or l2 is a minor 3rd above l1
            return True
    else:
        return False

def get_chord_relationship(l1, l2):
    if l1 == "silence" or l2 == "silence":
        return "silence"
    elif chords_are_related(l1, l2):
        return "related"
    else:
        return "unrelated"

def get_most_common_in_list(l):
    max_frequency = 0
    most_common = l[0]
     
    for item in l:
        curr_frequency = l.count(item)
        if(curr_frequency > max_frequency):
            max_frequency = curr_frequency
            most_common = item
 
    return most_common