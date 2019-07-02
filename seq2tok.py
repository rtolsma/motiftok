import numpy as np
import csv
import pandas as pd
from motifs import Motif, Motifs
from dijk import get_ordered_data_with_annotations


def tokenize_annotations(motifs, importance_scores=None):
    """
     Generates motif tokenizations of sequences from the Dijk Annotations
     given a Motifs object and optional importance scores. 
    :return: Tokenized sequences (sequences without motifs will be mapped to generic one hots)
    """

    ## Build Motif List and get data
    X, Y, annotations = get_ordered_data_with_annotations()

    # One Hot Encode the Motif data
    names = motifs.motif_names + [None]
    motif_dict = {k: [1 if i==j else 0 for j in range(len(names))] for i, k in enumerate(names)}
    motif_decoder = {str(motif_dict[k]): k for k in motif_dict}
    

    tokenized_seqs = []
    #offset = 0
    for i in range(len(annotations)):
        tokenized_seq = []
        matches = motifs.get_matches(i)
        for motif in matches:
            # TODO: experiment with other windows for relative positions
            # Fix alignment of motif names
            if importance_scores:
                #affinity = np.sum(np.abs(model_scores)[i-offset, motif.start:motif.end, :])/np.sum(np.abs(model_scores)[i-offset,:,:])
                affinity = np.sum(np.abs(importance_scores)[i, motif.start:motif.end, :])/np.sum(np.abs(importance_scores)[i,:,:])
                tokenized_seq.append(motif_dict[motif.name] + [affinity] + [motif.start] + [motif.end-motif.start])
            else:
                tokenized_seq.append(motif_dict[motif.name] + [motif.start] + [motif.end-motif.start])

            
        # Only add seqs that are non-trivial motif-tokenizations
        if importance_scores:
            tokenized_seqs.append(tokenized_seq + [motif_dict[None] + [0] + [0] + [0]] * (motifs.max_len - len(tokenized_seq)))
        else:
            tokenized_seqs.append(tokenized_seq + [motif_dict[None] + [0] + [0]] * (motifs.max_len - len(tokenized_seq)))

        
        
        #else:
        #    offset += 1
    
    # vectorize the sequences, (motif_names + (2 or 3), max_len) size with the transpose now...
    tokenized_seqs = np.stack([np.array(seq).T for seq in tokenized_seqs], axis=0)
    return tokenized_seqs, Y, motif_decoder