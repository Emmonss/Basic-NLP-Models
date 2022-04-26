import sys
sys.path.append('../')

from Segment.HMM.make_matrix_file import Make_HMM_Matrx


class Vertebi():
    def __init__(self,mode='train',**kwargs):
        hmm_matrix = Make_HMM_Matrx()

    def _forward(self):
        pass

    def _get_max_pre(self):
        pass

    def _backword(self):
        pass

    def get_segment_tags(self):
        pass