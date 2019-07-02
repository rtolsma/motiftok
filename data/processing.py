import csv
import pickle
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

#TODO: add save data if not saved

def get_tsv_iter(tsv):
    with open(tsv, 'r') as tsv_in:
        return list(csv.reader(tsv_in, delimiter='\t'))


def one_hot_encode(seq):
    """
    Get one hot encoding as 2D np array from sequence
    :param seq:
    :return: 4 x n one hot coded array
    """
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([encoding[n] for n in seq])

def one_hot_decode(seq):
    """
    Get one hot encoding as 2D np array from sequence
    :param seq:
    :return: 4 x n one hot coded array
    """
    encoding = {'[[1.0], [0.0], [0.0], [0.0]]': 'A',
                '[[0.0], [1.0], [0.0], [0.0]]': 'C',
                '[[0.0], [0.0], [1.0], [0.0]]': 'G',
                '[[0.0], [0.0], [0.0], [1.0]]': 'T'}
    return [encoding[str(n)] for n in seq]


def complement(seq):
    comp_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return list(map(comp_map.get, list(seq)))


def reverse_complement(s):
    return complement(s[::-1])


class Data:
    def __init__(self, data_set, seed=1):
        self.data_set = data_set
        self.X, self.Y = self.load_processed_data()
        self.W, self.type_key, self.subtype_key, self.strength_key, self.orientation_key = self.load_words()
        self.Y_mean = self.Y.mean(axis=0)
        self.Y_std = self.Y.std(axis=0)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.1, random_state=seed)

    def load_processed_data(self):
        # load processed data from file
        with h5py.File('data/{0}/seq_data.h5'.format(self.data_set), 'r') as hf:
            X = hf['X'][:]
            X = np.reshape(X, np.shape(X) + (1,))
            return X, hf['Y'][:]

    def load_words(self):
        # load processed data from file
        with open('data/{0}/keys'.format(self.data_set), 'rb') as key_file:
            type_key, subtype_key, strength_key, orientation_key = pickle.load(key_file)
        with h5py.File('data/{0}/seq_data.h5'.format(self.data_set), 'r') as hf:
            return hf['words'][:], type_key, subtype_key, strength_key, orientation_key


