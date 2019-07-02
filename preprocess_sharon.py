import csv, re, argparse
import torch
import pickle

def get_tsv_iter(tsv):
    with open(tsv, 'r') as tsv_in:
        return list(csv.reader(tsv_in, delimiter='\t'))

def process_elements(data):
    '''Processes sequences and extracts motifs with sequences.
    :param: dataset (list of lines in file)
    :return: array of sequences, set of elements in each sequence
    (element, start, end), list of unique elements
    '''
    seqs, elsets, elnames = [],[],[]
    for line in data:
        id, element_str, seq, numReads, expression = line
        elements = element_str.split('|')
        elset  = []
        for element in elements:
            # if want tf only uncomment next line
            #if "TF" not in element: continue
            if "Name=" not in element or "Sequence=" not in element: continue
            eldict = dict((k.strip(), v.strip()) for k,v in
              (item.split('=') for item in element.split(',')))
            matches = [(eldict['Name'], m.start(0), m.end(0)) for m in re.finditer(eldict['Sequence'], seq)]
            elset += matches
            elnames += [eldict['Name']]
        seqs += [seq]
        elsets += [set(elset)]
    return seqs, elsets, set(elnames)

def one_hot_encode_seqs(seqs, element_lists, element_names):
    '''one hot encode sequences - index A,C,G,T and all elements, and then one hot encode sequences
       :param: seqs: {list} of sequences,
               element_lists: {list} of elements in every sequence,
               element_names: {list} of element names
       :return: output data tensor: [num_seqs, len_seq,num_tokens], dict from token to index
   '''
    all_tokens = ['A','C','G','T', 'EOS'] + list(element_names)
    token_dict = dict((token, i) for i, token in enumerate(all_tokens))
    out_data = torch.zeros((len(seqs), len(seqs[0]), len(all_tokens) ))
    for i, seq in enumerate(seqs):
        for j, char in enumerate(seq):
            out_data[i, j, token_dict[char]] = 1
        for el, start, end in element_lists[i]:
            out_data[i, start:end, token_dict[el]] = 1
    return out_data, token_dict


def main():
    parser = argparse.ArgumentParser(description='Preprocessing Van Dijk data.')
    parser.add_argument("--datadir", default='./data/', help="Data directory")
    parser.add_argument("--dataset", default='sharon_promoters_w_tf.tsv', help='Original TSV file')
    parser.add_argument("--output", default='sharon_data_processed.p', help="Output file name")
    args = parser.parse_args()
    data = get_tsv_iter(args.datadir + args.dataset)
    seqs, elements, element_names = process_elements(data)
    indexed_data, token_dict = one_hot_encode_seqs(seqs, elements, element_names)
    with open(args.datadir + args.output, 'wb') as f:
        pickle.dump(indexed_data, f)
        pickle.dump(token_dict, f)

if __name__ == '__main__':
    main()
