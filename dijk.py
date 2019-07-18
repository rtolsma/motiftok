import numpy as np
import h5py
import pickle
import json
import re

from itertools import chain
from data.processing import *


def load_X():
    """
    Creates dict of LibID to data
    :return:
    """
    return {int(float(x[1])): [one_hot_encode(x[3]), one_hot_encode(reverse_complement(x[3]))]
            for x in get_tsv_iter('data/dijk/GSE92306_annotation_all_constructs.tsv')[1:]}


def load_seqs():
    return [x[3] for x in get_tsv_iter('data/dijk/GSE92306_annotation_all_constructs.tsv')[1:]]


def transform_y(y, log_plus=True):
    """
    transforms the Y data for learning, converts string '' to NaN
    :param y: the input Y array
    :param log_plus: whether to logplus transform the data
    :return: transformed Y data
    """
    y = np.array([float(n) if n != '' else float('nan') for n in y])
    if log_plus:
        # takes y out of the log2 transform given in original dataset
        y = np.power(2, y)
        # put y into log(1+y) to remove negative/very small float values -- avoid error
        y = np.log(1+y)
    # Issue here is that there's a lot of missing data, especially in the variance data, cols 7-12; only take the much
    # less sparse expression data for now
    return y[0:6]

def load_original_Y():
    """
    Creates dict of LibID to data without transforming Y
    :return:
    """
    return {int(float(y[-1])): transform_y(y[:-1], log_plus=False)
            for y in get_tsv_iter('data/dijk/GSE92306_expression_all_constructs.tsv')[1:]}

def load_Y():
    """
    Creates dict of LibID to data with transforming Y
    :return:
    """
    return {int(float(y[-1])): transform_y(y[:-1])
            for y in get_tsv_iter('data/dijk/GSE92306_expression_all_constructs.tsv')[1:]}


def load_annotations():
    raw_annotations = get_tsv_iter('data/dijk/GSE92306_annotation_all_constructs.tsv')[1:]
    annotations = {}
    for a in raw_annotations:
        annotation_info = re.split('[|#]', a[4])
        elements = []
        for key, val in zip(annotation_info[0::2], annotation_info[1::2]):
            if key == 'Element':
                vals = [(v.split('=')[0], v.split('=')[1]) for v in val.split(',')]
                element_dict = dict(vals)
                elements.append(element_dict)
        annotation = {'SetName': a[2], 'Sequence': a[3], 'Context': annotation_info[3].split('=')[1]}
        annotation.update({'Elements': elements})
        annotations.update({int(float(a[1])): annotation})
    return annotations

def remove_nan(Y):
    # if any of the labeled values are nan, the example is not usable, do not add it to the set; this misses out
    # 1678 examples that have at least 1 nan value
    for k in list(Y.keys()):
        if any(np.isnan(Y[k])):
            del Y[k]
    return Y

def get_ordered_data(rev_comp=False):
    """
    Creates ordered lists of X, Y on the keys from the dicts generated in loading X, Y
    :return: numpy arraya of ordered X, Y
    """
    X = load_X()
    Y = remove_nan(load_Y())
    # some data is missing, and data is unordered, so find ids common to x and y and create ordered lists of matching
    # examples associated with labels
    usable_indices = list(set.intersection(set(Y.keys()), set(X.keys())))
    nX, nY = [], []
    for i in usable_indices:
        if rev_comp:
            nX += X[i]
            nY += [Y[i], Y[i]]
        else:
            nX.append(X[i][0])
            nY.append(Y[i])
    X = np.array(nX)
    X = np.reshape(X, np.shape(X) + (1,))
    Y = np.array(nY)
    return X, Y

def get_ordered_data_with_annotations():
    """
    Loads all data for use in visulaizations
    :return: X, Y, and annotaions
    """
    annotations = load_annotations()
    X = load_X()
    Y = remove_nan(load_Y()) # Instead log_transform+remove nan load_original_Y()
    # some data is missing, and data is unordered, so find ids common to x and y and create ordered lists of matching
    # examples associated with labels
    usable_indices = list(set.intersection(set(annotations.keys()), set(Y.keys()), set(X.keys())))
    annotations = np.array([annotations[i] for i in usable_indices])
    X = np.array(list(chain([X[i] for i in usable_indices])))
    X = np.reshape(X, np.shape(X) + (1,))
    Y = np.array([Y[i] for i in usable_indices])
    return X, Y, annotations


def get_element_field_list(key, elements):
    return [e[key] for e in elements]


def get_tf_data(X, Y, annotations, tf='', tf_count=-1, weak_strong='both', exclude_non_tf=True, polyT_count=-1):
    """
    Gets specific subsets of filtered data for use in visualization scripts
    :param X: input one-hot encoded data
    :param Y: output labels
    :param annotations: list of dicts of annotation data
    :param tf: the string of the tf of interest, blacnk for all tfs
        Possible tf strings: GCN4, GAL4, LEU3, BAS1, MET31, LYS14, UGA3, SWI4
    :param tf_count: restricts to set number of tf binding sites of the given type, leave -1 for all
    :param weak_strong: whether to filter for 'Weak' or 'Strong', or 'mix' strength elements, leave 'both' for both
    :param exclude_non_tf: whether to allow non-tf sequence in the returned subset of the data
    :param polyT_count: filter for given number of polyT elements, leave -1 for all, only works if exclude_non_tf=False
    :return: subsets of X, Y, annotations
    """
    idx = []
    for i in range(len(annotations)):
        a = annotations[i]
        elem_types = get_element_field_list('SubType', a['Elements'])
        if tf in str(elem_types):
            if exclude_non_tf:
                if len(set(elem_types)) != 1:
                    continue
            if tf_count != -1:
                if str(elem_types).count(tf) != tf_count:
                    continue
            if weak_strong != 'both':
                if weak_strong == 'mix':
                    print([elem['Strength'] for elem in a['Elements'] if tf in elem['SubType']])
                    if not (any([elem['Strength'] == 'Strong' for elem in a['Elements'] if tf in elem['SubType']]) and
                            any([elem['Strength'] == 'Weak' for elem in a['Elements'] if tf in elem['SubType']])):
                        continue
                elif any([elem['Strength'] != weak_strong for elem in a['Elements'] if tf in elem['SubType']]):
                    continue
            if polyT_count != -1:
                if str(elem_types).count('polyT') != polyT_count:
                    continue
            idx.append(i)
    annotations = [annotations[i] for i in idx]
    X = X[idx, :, :, :]
    Y = Y[idx, :]
    return X, Y, annotations


def build_motif_list(out_file='motif_list.json'):
    annotations = load_annotations()
    motifs = {}
    for a in annotations.values():
        for e in a["Elements"]:
            motifs.update({e["Name"]: e["Sequence"]})
    with open("data/dijk/" + out_file, 'w') as out_file:
        json.dump(motifs, out_file)

def save_processed_data():
    X, Y = get_ordered_data(rev_comp=False)
    # save processed data to file
    with h5py.File('data/dijk/seq_data.h5', 'w') as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)

def save_as_words():
    X, Y, a = get_ordered_data_with_annotations()
    ds = []
    for s in a:
        elems = []
        for e in s['Elements']:
            elem = [e['Type'], e['SubType'], e['Strength'], e['Orientation'], e['Start'], e['End']]            
            elems.append(elem)
        elems += (7 - len(elems)) * [[-1, -1, -1, -1, -1, -1]]
        # 7 is hardcoded because it is the max # motifs in the data set
        ds.append(elems)

    npds = np.array(ds)

    type_key = {i: k for i, k in enumerate(filter(lambda x: x != '-1', np.unique(npds[:, :, 0])))}
    subtype_key = {i: k for i, k in enumerate(filter(lambda x: x != '-1', np.unique(npds[:, :, 1])))}
    strength_key = {i: k for i, k in enumerate(filter(lambda x: x != '-1', np.unique(npds[:, :, 2])))}
    orientation_key = {i: k for i, k in enumerate(filter(lambda x: x != '-1', np.unique(npds[:, :, 3])))}

    for i in type_key:
        npds[type_key[i] == npds] = i

    for i in subtype_key:
        npds[subtype_key[i] == npds] = i

    for i in strength_key:
        npds[strength_key[i] == npds] = i

    for i in orientation_key:
        npds[orientation_key[i] == npds] = i

    npds = npds.astype(float)

    with h5py.File('data/dijk/seq_data.h5', 'w') as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)
        hf.create_dataset("words", data=npds)

    with open('data/dijk/keys', 'wb') as key_file:
        pickle.dump((type_key, subtype_key, strength_key, orientation_key), key_file)












