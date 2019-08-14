import numpy as np

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.layers import NonlinearMxtsMode


from importlib import reload
from deeplift.visualization import viz_sequence
import shap
import shap.explainers.deep.deep_tf


def one_hot_encode_along_row_axis(sequence):
    #theano dim ordering, uses row axis for one-hot
    to_return = np.zeros((1,4,len(sequence)), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0],
                                 sequence=sequence, one_hot_axis=0)
    return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1):
        assert zeros_array.shape[0] == len(sequence)
    #zeros_array should be an array of dim 4xlen(sequence), filled with zeros.
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1


from deeplift.dinuc_shuffle import dinuc_shuffle, traverse_edges, shuffle_edges, prepare_edges
from collections import Counter

def onehot_dinuc_shuffle(s):
    s = np.squeeze(s)
    argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
    #print("\n" + argmax_vals)
    #print('\n\n', traverse_edges(argmax_vals, shuffle_edges(prepare_edges(argmax_vals))))
    shuffled_argmax_vals = [int(x) for x in traverse_edges(argmax_vals,
                            shuffle_edges(prepare_edges(argmax_vals)))]
    to_return = np.zeros_like(s)
    to_return[list(range(len(s))), shuffled_argmax_vals] = 1
    #print('To Return:', to_return)
    return to_return

def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        #At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        #For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        #The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1)
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    return to_return

def get_hyp_scores(model, hot_seqs, targets=False):
    shuffle_several_times = lambda s: np.array([onehot_dinuc_shuffle(s) for _ in range(10)])
    dinuc_shuff_explainer = shap.DeepExplainer(
        (model.input, model.output), shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref)
    shap_explanations = dinuc_shuff_explainer.shap_values(hot_seqs)
    return shap_explanations
    for idx, orig_seq in enumerate(hot_seqs):
        print("Sequence idx", idx)
        print("Actual contributions")
        hypimpscores = shap_explanations[idx]
        # (The actual importance scores can be computed using an element-wise product of
        #  the hypothetical importance scores and the actual importance scores)
        print(len(shap_explanations))
        print(hypimpscores.shape, orig_seq.shape)
        viz_sequence.plot_weights((hypimpscores * orig_seq)[0], subticks_frequency=20)
        print("Hypothetical contributions")
        viz_sequence.plot_weights(hypimpscores, subticks_frequency=20)

    return shap_explanations

