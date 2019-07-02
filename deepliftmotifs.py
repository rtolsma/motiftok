import pandas as pd
import numpy as np
import json
import h5py

from seq2tok import tokenize_annotations
from dijk import get_ordered_data_with_annotations

from keras.models import (Sequential, load_model, model_from_json, model_from_yaml)
from keras.layers import (
    Dense, Dropout, Conv2D, Conv1D, Flatten, SpatialDropout1D, Lambda,
    MaxPool1D, LSTM, RNN, BatchNormalization, Input
)
import keras.regularizers as reg
import keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.visualization import viz_sequence
from deeplift.util import get_shuffle_seq_ref_function, get_hypothetical_contribs_func_onehot
from deeplift.dinuc_shuffle import dinuc_shuffle #function to do a dinucleotide shuffle

import modisco
import modisco.backend
import modisco.nearest_neighbors
import modisco.affinitymat
import modisco.tfmodisco_workflow.seqlets_to_patterns
from modisco.tfmodisco_workflow.workflow import TfModiscoWorkflow
import modisco.aggregator
import modisco.cluster
import modisco.value_provider
import modisco.core
import modisco.coordproducers
import modisco.metaclusterers


class Trainer:
    '''
    Design Decision: When comparing models that train on the full sequences versus
    the motif tokenizations, we need to be careful about the train/test splits
    to make sure that they have equally valid test data to measure from. Same goes
    for the ensemble models.

    This gets more difficult, since depending on which motifs we want to test,
    we should omit full sequences that don't match any motifs, as the tokenized model
    won't have those to train off of.
    '''
    def __init__(self, motifs):
        self.motifs = motifs
        X, Y, annotations = get_ordered_data_with_annotations()
        self.X = X[motifs.non_empty_indices, :, :, :, 0]
        self.Y = Y[motifs.non_empty_indices]
        self.annotations = annotations[motifs.non_empty_indices]
        indices = np.arange(len(motifs.non_empty_indices))
        np.random.shuffle(indices)
        split = int(0.8 * indices.shape[0])
        self.train_indices = indices[:split]
        self.test_indices = indices[split:]

    def trainFullSequence(self, seqmodel=None, epochs=50):    
        if seqmodel is None:
            seqmodel = Trainer.getModel()
        ### Full Sequence Training
        X_train, Y_train = self.X[self.train_indices], self.Y[self.train_indices]

        # 2nd axis stores reverse complement, so unfold that
        X_train_rev = np.concatenate((X_train[:,0,:,:], X_train[:,1,:,:]), axis=0)
        Y_train_rev = np.concatenate((Y_train, Y_train), axis=0)

        seqmodel.fit(X_train_rev, Y_train_rev, validation_split=0.2, batch_size=256, epochs=epochs)
        return seqmodel
    

    def trainTokenized(self, tokmodel=None, importance_scores=None, epochs=50):
        tokenizedX, Y, motif_decoder = tokenize_annotations(self.motifs, importance_scores)
        
        if tokmodel is None:
            tokmodel = Trainer.getModel(input_shape=tokenizedX.shape[1:])

        # take out the sequences without motif matches, same as we did for the full sequences
        tokenizedX, Y = tokenizedX[self.motifs.non_empty_indices], Y[self.motifs.non_empty_indices]
        tokenizedX_train, Y_train = tokenizedX[self.train_indices], Y[self.train_indices]
        tokmodel.fit(tokenizedX_train, Y_train, validation_split=0.2, batch_size=256, epochs=epochs)

        return tokmodel

    def trainFullSequenceEnsemble(self, models=None, N=10, epochs=50):
        if not models:
            models = [Trainer.getModel() for i in range(N)]
        
        ensemble_scores = []
        for m in models:
            _ = self.trainFullSequence(m, epochs)
            score = self.getScores(m)
            ensemble_scores.append(score)
        

        return models, ensemble_scores
        
    def compareExperiments(self, seqmodel, tokmodel, importance_scores=None):
        X_test, Y_test = self.X[self.test_indices], self.Y[self.test_indices]
        tokenizedX, _, motif_decoder = tokenize_annotations(self.motifs, importance_scores)
        tokenizedX_test = tokenizedX[self.test_indices]
        seqpreds = seqmodel.predict(X_test[:, 0, :, :]) # only use one side of the complements
        tokpreds = tokmodel.predict(tokenizedX_test)
        
        seq_corr = r2_score(seqpreds, Y_test)
        tok_corr = r2_score(tokpreds, Y_test)
        model_corr = r2_score(seqpreds, tokpreds)
        
        print('Seqmodel Correlations: ', seq_corr)
        print('Tokmodel Correlations: ', tok_corr)
        print("Model correlations : ", model_corr)


    @staticmethod
    def getModel(regularization=True, input_shape=(150,4), num_units=1):
        # play around with hyperparameters?? seems like a fine model so far tho
        s = 7
        w = 10
        p = 0.5
        l = 10
        model = Sequential()
        model.add(Conv1D(filters=s, kernel_size=(w, ), padding='same',activation='relu', input_shape=input_shape))

        def addConvUnits(model, reg):
            model.add(Conv1D(filters=s, kernel_size=(w, ), padding='same',activation='relu'))
            model.add(Conv1D(filters=s, kernel_size=(w, ), padding='same',activation='relu'))
            model.add(Conv1D(filters=s, kernel_size=(w, ), padding='same',activation='relu'))
            model.add(Conv1D(filters=s, kernel_size=(w, ), padding='same',activation='relu'))

            if reg:
                # model.add(BatchNormalization()) Probs not useful
                model.add(Dropout(p))
        for _ in range(num_units):
            addConvUnits(model, regularization)
            
        model.add(Flatten())
        model.add(Dense(l, activation='relu'))
        model.add(Dense(6))
        model.compile(optimizer='adam', loss='mse')
        return model


    def getScores(self, seqmodel, temp_path='./models/tempscores.h5'):
        seqmodel.save(temp_path)
        return self.getDeepliftScores(temp_path, None)

    ## Deeplift Scoring Visualization
    def plotDeepLift(self, importance_scores, num=20, verbose=False):
        for i in range(importance_scores.shape[0]):
            if i > num:
                break
            highlights = {}
            highlights['b'] = [(m.start, m.end) for m in self.motifs.get_matches(i)]
            viz_sequence.plot_weights(importance_scores[i], highlight=highlights)
            if verbose:
                for j, m in enumerate(self.motifs.get_matches(i)):
                    sub_seq = m.matched
                    print(f'{j}:', sub_seq)

    ### Setup Deeplift for affinity scoring, only for models trained on full sequences
    def getDeepliftScores(self, weight_path, yaml_path, data=None):
        if data is None:
            data = self.X[:, 0, :, :]

        deeplift_model = kc.convert_model_from_saved_files(
            weight_path,
            yaml_path,
            nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault
            )
        deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                                    find_scores_layer_idx=0,
                                    target_layer_idx=-1)

        scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[data],
                                         batch_size=50,
                                         progress_update=4000))
        '''
        contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,
                                                        target_layer_idx=-1)
        contribs_many_refs_func = get_shuffle_seq_ref_function(
            score_computation_function=contribs_func,
            shuffle_func=dinuc_shuffle)
        '''

        multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0,
                                                                    target_layer_idx=-1)
        hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(multipliers_func)

        #Once again, we rely on multiple shuffled references
        hypothetical_contribs_many_refs_func = get_shuffle_seq_ref_function(
            score_computation_function=hypothetical_contribs_func,
            shuffle_func=dinuc_shuffle)
        #idk??
        num_refs_per_seq = 10
        hypothetical_scores = hypothetical_contribs_many_refs_func(
                                task_idx=0,
                                input_data_sequences=data,
                                num_refs_per_seq=num_refs_per_seq,
                                batch_size=50,
                                progress_update=1000,
                            )
        # mean normalize?
        hypothetical_scores = hypothetical_scores - np.mean(hypothetical_scores, axis=-1)[:,:,None]
        
        return scores, hypothetical_scores

    def tfmodiscoResults(self, scores, hypothetical_scores):
        task_to_scores, task_to_hyp_scores = {'task0':scores}, {'task0': hypothetical_scores}
        tfmodisco_results = TfModiscoWorkflow(seqlets_to_patterns_factory=
                                                modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                                                    #trim_to_window_size=15,
                                                    #initial_flank_to_add=5,
                                                    #kmer_len=5, num_gaps=1,
                                                    #num_mismatches=0,
                                                    final_min_cluster_size=20)
                                            )(task_names=['task0'], 
                                                contrib_scores=task_to_scores,
                                                hypothetical_contribs=task_to_hyp_scores,
                                                one_hot=self.X[:, 0,:,:])
        return tfmodisco_results

