"""
Interface to viterbi C-code
"""

import numpy
import ctypes
import unittest
import struct 

try:
    ## Load shared library
    _libviterbi = numpy.ctypeslib.load_library('libviterbi',__file__)
    
    ## Specify the library signature
    _libviterbi.viterbi.argtypes = [ctypes.c_int, 
                                    numpy.ctypeslib.ndpointer(dtype = numpy.int32),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    numpy.ctypeslib.ndpointer(dtype = numpy.double),
                                    numpy.ctypeslib.ndpointer(dtype = numpy.double),
                                    numpy.ctypeslib.ndpointer(dtype = numpy.double),
                                    ctypes.c_int,
                                    numpy.ctypeslib.ndpointer(dtype = numpy.int32),
                                    numpy.ctypeslib.ndpointer(dtype = numpy.int32)]
    
    ## Specify return type
    _libviterbi.viterbi.restype = ctypes.c_void_p
except ImportError:
    _libviterbi = None

def viterbi(seq, Pij, P, Pi, lead_vec=[], chk = 0):
    """
    Calculate the viterbi sequence of x given the hmm specified by Pij, P, Pi
    Pij: m x m numpy array, transition prob from state i to state j
    P:   m x a numpy array, emission prob from state i of symbol j
    Pi:  m x 1 numpy array, initial prob
    lead_vec: vector containing sparse encoding of which states lead to state i
    
    Example:
    import numpy
    import viterbi
    x = numpy.array([1, 0, 0, 1, 1, 1, 0])
    Pij = numpy.array([[0.7, 0.2, 0.1], [0.2, 0.8, 0.0], [0.1, 0.0, 0.9]])
    P = numpy.array([[0.9, 0.1],[0.5, 0.5],[0.2, 0.8]])
    Pi = numpy.array([0.5, 0.0, 0.5])
    lead_vec = numpy.array([3, 0, 1, 2, 2, 0, 1, 2, 0, 2])
    viterbi.viterbi(x, Pij, P, Pi) 
    """
 
    ## Convert args to numpy arrays
    seq = numpy.asarray(seq, dtype=numpy.int32)
    Pij = numpy.asarray(Pij, dtype=numpy.double)
    P = numpy.asarray(P, dtype=numpy.double)
    Pi = numpy.asarray(Pi, dtype=numpy.double)
    lead_vec = numpy.asarray(lead_vec, dtype=numpy.int32)
        
    ## Check the args
    if chk:
        if Pij.ndim != 2 or P.ndim != 2 or Pi.ndim != 1:
            raise ValueError("Bad shape in args")
        if Pij.shape[0] != Pij.shape[1] or Pij.shape[0] != P.shape[0] or Pij.shape[0] != len(Pi):
            raise ValueError("Bad dimension in args")
        if seq.min() < 0 or seq.max() > Pij.shape[0]-1:
            raise ValueError("Bad values in seq")
        if len(lead_vec):
            at = 0
            for j in range(Pij.shape[0]):
                if at >= len(lead_vec):
                    raise ValueError("Bad lead_vec")
                at += 1
                for i in range(lead_vec[j]):  
                    if at >= len(lead_vec) or lead_vec[at] < 0 or lead_vec[at] >= Pij.shape[0]: 
                        raise ValueError("Bad lead_vec")
                    at += 1
        
    
    ## Allocate result
    vit = numpy.empty(len(seq), dtype=numpy.int32)
    _libviterbi.viterbi(ctypes.c_int(len(seq)), seq, ctypes.c_int(P.shape[0]), ctypes.c_int(P.shape[1]), Pi, Pij, P, ctypes.c_int(len(lead_vec)), lead_vec, vit)
    # _libviterbi.viterbi(len(seq), seq, P.shape[0], P.shape[1], Pi, Pij, P, len(lead_vec), lead_vec, vit)
    
    return vit

class TestViterbi(unittest.TestCase):
    """ Test the viterbi function """
    
    def setUp(self):
        """ Set some random-y parameters """
        self.Pij = numpy.array([[0.7, 0.2, 0.1], [0.2, 0.8, 0.0], [0.4, 0.0, 0.6]])
        self.P = numpy.array([[0.9, 0.1],[0.5, 0.5],[0.2, 0.8]])
        self.Pi = numpy.array([0.5, 0.0, 0.5])
        self.lead_vec = numpy.array([3, 0, 1, 2, 2, 0, 1, 2, 0, 2])

    def testViterbiNoLead(self):
        """ Test without lead_vec """
        v = viterbi(numpy.array([1, 0, 0, 1, 0, 1, 1, 0]), self.Pij, self.P, self.Pi) 
        self.assert_(all(v == numpy.array([2, 0, 0, 1, 1, 1, 1, 1])))
        
    def testViterbiLead(self):
        """ Test using lead_vec """
        v = viterbi(numpy.array([1, 0, 0, 1, 0, 1, 1, 0]), self.Pij, self.P, self.Pi, self.lead_vec) 
        self.assert_(all(v == numpy.array([2, 0, 0, 1, 1, 1, 1, 1])))
    
""" Run test if run """
if __name__ == '__main__':
    unittest.main()
