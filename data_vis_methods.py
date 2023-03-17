import networkx as nx
import matplotlib.pyplot as plt
from helpers import split_label, get_chord_relationship, get_all_labels, get_short_label
from constants import *
import seaborn as sns
import os

def calculate_node_position(label):
    if label == "silence":
        return [0, -1]
    note, octave, chord_type = split_label(label)

    note_offset = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "Ab": 8,
        "A": 9,
        "Bb": 10,
        "B": 11,
    }

    x = ((int(octave) - 4) * len(notes) + note_offset[note]) * 2
    y = 1 if chord_type == "major" else 0
    return [x,y]

def get_edge_colour(l1, l2):
    relationship = get_chord_relationship(l1, l2)
    if relationship == "silence":
        return "blue"
    elif relationship == "related":
        return "green"
    else:
        return "red"


def generate_missclassification_graph(actual_labels, predicted_labels, feature_type, model_type, accuracy):
    G = nx.DiGraph()

    all_labels = get_all_labels()
    for label in all_labels:
        G.add_node(label, pos=calculate_node_position(label), label=get_short_label(label))

    weights = {}
    for l1 in all_labels:
        weight_list = {}
        for l2 in all_labels:
            weight_list[l2] = 0
        weights[l1] = weight_list

    for (actual, predicted) in zip(actual_labels, predicted_labels):
        if not actual == predicted:
            weights[actual][predicted] += 1

    edge_weights = []
    edge_colours = []

    for l1 in all_labels:
        for l2 in all_labels:
            weight = weights[l1][l2]
            if not weight == 0:
                G.add_edge(l1, l2)
                edge_weights.append(weight / 4)
                edge_colours.append(get_edge_colour(l1, l2))


    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    plt.suptitle(f"Misclassification graph - {feature_type} {model_type}", fontsize=7)
    plt.title(f"Accuracy: {accuracy}", fontsize=5)
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=150, font_size=3, width=edge_weights, edge_color=edge_colours, arrowsize=7)
    
    path = f"plots/misclassification_graphs"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/{feature_type}-{model_type}.png", format="PNG", dpi=1000)
    plt.clf()

def make_accuracy_heatmap(accuracies):
    ax = plt.axes()
    sns.heatmap(accuracies, annot=True, annot_kws={"size": 8}, yticklabels=model_types, xticklabels=feature_types, cmap=sns.cm.rocket_r)
    ax.set_title("Model Accuracy")
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(rotation=45, fontsize=8)
    path = "plots"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/accuracy_heatmap.png", format="PNG", dpi=1000)
    plt.clf()

def tally_misclassification_types(actual_labels, predicted_labels):
    num_related = 0
    num_unrelated = 0
    num_silence = 0
    for (l1, l2) in zip(actual_labels, predicted_labels):
        if not l1 == l2:
            relationship = get_chord_relationship(l1, l2)
            if relationship == "related":
                num_related += 1
            elif relationship == "unrelated":
                num_unrelated += 1
            elif relationship == "silence":
                num_silence += 1
            else:
                print("We got issues in tally_misclassification_types")
                exit()
    return num_related, num_unrelated, num_silence

def make_misclassification_bar_graph(model_types, feature_types, related_vals, unrelated_vals, silence_vals):
    fig, axs = plt.subplots(len(model_types), 1)
    plt.xticks(fontsize=6, rotation=30)
    for model_idx, model_type in enumerate(model_types):
        axs[model_idx].bar(feature_types, unrelated_vals[model_idx, :], label="Unrelated", color="red")
        axs[model_idx].bar(feature_types, related_vals[model_idx, :], bottom=unrelated_vals[model_idx, :], label="Related", color="green")
        axs[model_idx].bar(feature_types, silence_vals[model_idx, :], bottom=unrelated_vals[model_idx, :] + related_vals[model_idx, :], label="Silence", color="blue")
        axs[model_idx].set_ylabel(model_type)
    fig.suptitle("Misclassification Types")
    plt.legend()
    path = "plots"
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(f"{path}/misclassification_bar_plot.png", format="PNG", dpi=1000)
    plt.clf()