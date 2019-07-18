import pandas as pd
import numpy as np
from dijk import get_ordered_data_with_annotations

from data.processing import one_hot_encode

## Everything is aligned with Dijk Annotations from get_ordered_annotations
## ...keep track of missing indices from empty motif matches
## Make a Standard Motif class
## standardize with Motifs class (Dijk + FIMO) for now
## seq2tok: sequences + motif=> position iterator + optional scoring
## deeplift viz: scores + motif=> position iterator
## Deep Learning: Decompose better (separate tok+seq)===> get_models, train_model, train_ensemble, compare_models, getXY, getTokXY

class Motif:
    '''
    Standardize the way we interact we motif taggings from different sources,
    and only keep most relevant data. Can add/subclass later
    '''

    def __init__(self, start, end, sequence, name, affinity=0):
        self.start = start
        self.end = end
        self.affinity = affinity
        self.sequence = sequence
        self.name = name
        self.matched = sequence[start:end+1]

    def __str__(self):
        return f'Name: {self.name} Match: ({self.start}, {self.end})'

class Motifs:
    '''
    Maintains a working collection of motifs and provides a uniform
    way of accessing individual motif matches to the van dijk sequences.

    motifs: List of List of Motif Objects
    '''
    def __init__(self, motifs):
        self.motifs = motifs
        self.non_empty_indices = [i for i,matches in enumerate(motifs) if matches]
        self.max_len = max([len(matches) for matches in motifs])
        self.motif_names = list(set([m.name for matches in motifs for m in matches]))

    def get_matches(self, index):
        '''
        index: Index corresponding to the row in the van dijk ordered_annotations
        Returns: List of Motif objects corresponding to that sequence
        '''
        return self.motifs[index]

    def get_positions(self, index):
        '''
        Returns: an iterator through the (start, end) tuples of the motif list
        '''
        return list(map(lambda x: (x.start, x.end) , self.motifs[index]))


    def get_references(self, length=150):
        '''
        Returns: list of list of indices + ref one-hot encode list
        '''
        # TODO: Fix...should'nt be hardcoded...
        return [self.non_empty_indices], [np.zeros((length, 4))]

class StubMotifs(Motifs):
    def __init__(self, length):
        motifs = [[]] * length
        super().__init__(motifs)
        self.non_empty_indices = list(range(length))


class DijkMotifs(Motifs):
    
    REFS = {#'context_PHO5_NATIVE': 'GGGGACCAGGTGCCGTAAGAATCCGGTACCACGTTTTCGCATAGAACGCAACTGCACAATGCCAAAAAAAGTAAAAGTGATTAAAAGAGTTAATCTCTAAATGAATCGATACAACCTTGGCACTCACACGTGGCGATCCTAGGGCGATCA',
            'context_TSA1_NATIVE': 'GGGGACCAGGTGCCGTAAGGCTCGCATATGTTCTGGCCCGTCGGGTTTTCTGACAAATTGTCCTTTAGGGATTTTTCGGTTTGGCTCGGGTTGGCAAAGTCGGCTGGCAACAAACCAGGACATATATAAAGGGCGATCCTAGGGCGATCA',
            'context_RPL3_10_NATIVE': 'GGGGACCAGGTGCCGTAAGACCGAAAGTACACAACTGTTTTCCATTTTTTTTTTTTTTTTTTCAGTGATCATCGTCCATGAAAAAAATTTTTCATTTGTCTCTTTCGTGCTTCCTGGATATATAAAATACGAGCGATCCTAGGGCGATCA' ,
            'context_HIS3_NATIVE': 'GGGGACCAGGTGCCGTAAGATCGGACCACTAGAGCTTGACGGGGAAAGCCGGCGAACGTGGCGAGAAAGGAAGGGAAGAAAGCGTTTTCATTTTTTTTTTTCCACCTAGCGGATGACTCTTTTTTTTTCTTAGCGATCCTAGGGCGATCA',
            'context_GAL1_10_NATIVE': 'GGGGACCAGGTGCCGTAAGGATACACTAACGGATTAGAAGCCGCCGAGCGGGCGACAGCCCTCCGACGGAAGACTCTCCTCCGTGCGTCCTCGTCTTCGAAACGCAGATGTGCCTCGCGCCGCACTGCTCCGGCGATCCTAGGGCGATCA'
            }

    def __init__(self):
        _, _, self.annotations = get_ordered_data_with_annotations()
        motifs, references = DijkMotifs.convert_dijk_motifs(self.annotations)
        super().__init__(motifs)
        # hax for testing rn
        self.non_empty_indices = list(range(len(self.motifs)))
        self.references = references

    def get_references(self):
        data_list, refs_list = [], []
        for k,v in DijkMotifs.REFS.items():
            data_list.append(self.references[k])
            refs_list.append(one_hot_encode(v))
        return data_list, refs_list

    @classmethod
    def convert_dijk_motifs(self, annotations):

        motifs = []
        references = {} # dict of contexts => indices
        for i,s in enumerate(annotations):
            matches = []
            seq = annotations[i]['Sequence']
            context = annotations[i]['Context'].replace('NULL', 'NATIVE')
            if seq in DijkMotifs.REFS.values():
                continue
            if references.get(context, None) is None:
                references[context] = []
            references[context].append(i)


            for e in s['Elements']:
                end_pos = int(float(e['End']))
                start_pos = int(float(e['Start']))
                name = DijkMotifs.convertMotifName(e['Name'])
                # the triplet NNN mutations aren't useful??
                #if name.startswith('triplet'):
                #    continue

                # i think the start,ends were reflected somehow
                #start_pos, end_pos = 150 - end_pos, 150 - start_pos + 1

                matches.append(Motif(start_pos, end_pos, seq, name))
            motifs.append(matches)
        return motifs, references


    @classmethod
    def convertMotifName(self, name):
        '''
        Helper Function for converting motif names from the van Dijk
        dataset to the names contained in the Swiss Regulon yeast motif
        names. Some names output will not be in the Regulon database
        i.e native promoters or polya, but any van Dijk annotation 
        containing a real motif name will have the correct output
        '''
        name = name.lower()
        if (name.startswith('allpssms') 
            or name.startswith('giniger')
            or name.startswith('matalpha2')):
            return name.split('_')[1]
        elif name.startswith('native_right') or name.startswith('native_left'):
            pass
        elif '_' in name:
            name = name.split('_')[0]
        elif name.startswith('polyt5'):
            name = 'polyt5'
        
        return name.lower()




class FimoMotifs(Motifs):

    annotations = None

    def __init__(self, path='./Fimo_motifs.tsv'):
        if FimoMotifs.annotations is None:
            _,_, FimoMotifs.annotations = get_ordered_data_with_annotations()
        fimo_matches = FimoMotifs.load_fimo_dict(path)
        motifs = FimoMotifs.get_fimo_motifs(fimo_matches)
        super().__init__(motifs)
    

    # The fimo motifs are in a csv which are identified by the sequence they relate to
    @classmethod
    def get_fimo_motifs(cls, fimo_matches):
        motifs = []

        for i in range(len(cls.annotations)):
            seq = cls.annotations[i]['Sequence']
            motifs.append( [FimoMotifs.convert_fimo(f) for f in fimo_matches[seq]])
        return motifs


    @staticmethod
    def convert_fimo(fimo_motif):
        start = fimo_motif['start']
        stop = fimo_motif['stop']
        score = fimo_motif['score']
        name = fimo_motif['motif_alt_id']
        sequence = fimo_motif['seq']
        return Motif(start, stop, sequence, name, score)

    @classmethod
    def load_fimo_dict(self, path):
        motif_matches = pd.read_csv(path, delimiter=',')
        seq_dict = {}
        for i in range(motif_matches.shape[0]):
            row = motif_matches.iloc[i]
            if not seq_dict.get(row['seq'], None):
                seq_dict[row['seq']] = []
            seq_dict[row['seq']].append(row.to_dict())
        
        for k in seq_dict:
            seq_dict[k] = sorted(seq_dict[k], key=lambda x: x['p-value'])
        
        return seq_dict
