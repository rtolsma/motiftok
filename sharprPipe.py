
# coding: utf-8

# In[18]:


import h5py
import pandas as pd
import numpy as np
import os


from data.processing import one_hot_encode, reverse_complement

from deepliftmotifs import Trainer
from motifs import DijkMotifs, FimoMotifs, StubMotifs

import keras

from modisco.visualization import viz_sequence

from dijk import get_ordered_data_with_annotations
from collections import defaultdict as ddict

from tfhyp import get_hyp_scores
import matplotlib.pyplot as plt

import pickle


def modisco_things(tfmodisco):
    #get_ipython().system("rm 'tfmodiscoresults.hdf5'")
    os.remove('tfmodiscoresults.hdf5')
    grp = h5py.File('tfmodiscoresults.hdf5')
    tfmodisco.save_hdf5(grp)
    hdf5_results = h5py.File('tfmodiscoresults.hdf5', 'r')

    print("Metaclusters heatmap")
    import seaborn as sns
    activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                        np.array(
            [x[0] for x in sorted(
                    enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
                   key=lambda x: x[1])])]
    sns.heatmap(activity_patterns, center=0)
    plt.show()

    metacluster_names = [
        x.decode("utf-8") for x in 
        list(hdf5_results["metaclustering_results"]
             ["all_metacluster_names"][:])]

    ids = set()

    all_patterns = []
    for metacluster_name in metacluster_names:
#        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                       [metacluster_name])
#        print("activity pattern:",metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in 
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                                 ["patterns"]["all_pattern_names"][:])]
        if (len(all_pattern_names)==0):
            print("No motifs found for this activity pattern")
        for pattern_name in all_pattern_names:
#            print(metacluster_name, pattern_name)
#            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            #print(pattern["seqlets_and_alnmts"]["seqlets"].value)

            for s in pattern['seqlets_and_alnmts']['seqlets'].value:
                ids.add(int(s.decode().split(',')[0].split(':')[1]))
 #           background = np.array([0.27, 0.23, 0.23, 0.27])
 #           print("Task 0 hypothetical scores:")
 #           viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"])
 #           print("Task 0 actual importance scores:")
 #           viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"])
 #           print("onehot, fwd and rev:")
 #           viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
 #                                                           background=background)) 
 #           viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
 #                                                           background=background)) 
    return ids


def get_sharpr_data(size=-1):
    ### rip memory
    data = pd.read_csv('./data/sharpr/sharprFullDataMatrix.tsv', delimiter='\t').iloc[:size]
    
    # cols = ['hepg2_minp_avg_count', 'hepg2_sv40p_avg_count', 'hepg2_minp_avg_norm', 'hepg2_sv40p_avg_norm']
    cols = [c for c in data.columns if 'hepg2' in c and 'avg' not in c]
    indices = [i for i, s in enumerate(data['sequence'].values) if 'N' not in s]
    X_reg = np.stack([one_hot_encode(s) for s in data['sequence'].iloc[indices].values])
    X_comp = np.stack([one_hot_encode(reverse_complement(s)) for s in data['sequence'].iloc[indices].values])
    X = np.zeros((X_reg.shape[0], 2, X_reg.shape[1], X_reg.shape[2]))
    X[:, 0, :, :] = X_reg
    X[:, 1, :, :] = X_comp
    
    Y = data[cols].iloc[indices].values
    return X, Y

def process_hypothetical_scores(hyp_scores):
    modiscos = []
    ids = []
    for k in range(len(hyp_scores)):
        scores = hyp_scores[k] * t.X[:,0, :, :]
        # scores = scores.mean(axis=0)
        hypothetical_scores = hyp_scores[k]
        #t.plotDeepLift(scores, num=20, verbose=True)
        tfmodisco = t.tfmodiscoResults(scores, hypothetical_scores)
        modiscos.append(tfmodisco)
        id_list = modisco_things(tfmodisco)
        ids.append(id_list)

    with open('/motifvol/ids.pickle', 'wb') as f:
        pickle.dump(ids, f)

    with open('/motifvol/hyp_scores.pickle', 'wb') as f:
        pickle.dump(modiscos, f)

#pd.read_csv('./data/sharpr/sharprFullDataMatrix.tsv', delimiter='\t').iloc[:10].columns
#x,y = get_sharpr_data(size=-1)

X, Y = None, None
with h5py.File('./dragonn/train.hdf5','r') as hf:
    X , Y = hf['X']['sequence'][:], hf['Y']['output'][:]

#t = Trainer(StubMotifs(length=x.shape[0]), X=X, Y=Y)

seqmodel = None
with open('./dragonn/model.json', 'r') as f:
    seqmodel = keras.models.model_from_json(f.read())
seqmodel.load_weights('./dragonn/pretrained.hdf5')

#seqmodel = t.trainFullSequence(epochs=1, input_shape=(145,4), output_shape=8)
hyp_scores = get_hyp_scores(seqmodel, X)

process_hypothetical_scores(hyp_scores)
