#! /usr/bin/env python
# Copyright (c) 2011 LightKeeper Advisors LLC
# ANY REDISTRIBUTION OR COPYING OF THIS MATERIAL WITHOUT THE EXPRESS CONSENT
# OF LIGHTKEEPER IS PROHIBITED.
# All rights reserved.
"""
Generic implementation of a fast hidden Markov model.
"""
import sys
import re
import math
import numpy
import unittest
import copy
import viterbi

class Hmm:
    """ 
    A `Hidden Markov Model<http://http://en.wikipedia.org/wiki/Hidden_Markov_model>`_ or
    HMM at it simplest form is a dynamic bayesian network.  It is commonly used
    in recognition or classification problems.
    """
    
    def __init__(self, start_spec, symbol_spec, state_spec):
        """
        start_spec:  A dictionary of state: probability initial states
        symbol_spec: A list of allowable tokens
        state_spec:  A list of (state_name, transition_dic, emission_dic) triplets
            transition_dic: A dictionary of {state_pattern: probability} pairs where state pattern can be a regex
            emission_dic: A dictionary of {symobl_pattern: probability} pairs where symbol pattern can be a regex
            
        Example:
        symbol_spec = ["uc", "lc", "ws"]
        state_spec = [("word",     {"word": 10, "space": 1}, {"uc":1, "lc":100}),
                      ("acronym",  {"acronym": 10, "space": 1}, {"uc":100}),
                      ("space",    {"word": 1, "acronym": 1, "space": 10}, {"ws":1})]
        

        """
        
        # Save spec's
        self._start_spec = copy.deepcopy(start_spec)
        self._symbol_spec = copy.deepcopy(symbol_spec)
        self._state_spec = copy.deepcopy(state_spec)
                
        # Working structures M = number of states, K = number of symbols
        self._Pij = []                   # Transition probabilities   (M x M)
        self._Pe  = []                   # State emission probabilies (M x K)
        self._Pi  = []                   # Initial state distribution (M x 1)  
        self._state_dic = {}             # State name to state index map
        self._symbol_dic = {}            # Symbol name to symbol index map
        self._state_rev = {}             # State index to state name map
        self._symbol_rev = {}            # Symbol index to symbol name map
        self._lead_state = []            # States that can lead to a state
        
        # Convert state_spec into useful quantities
        self._M = len(self._state_spec)
        self._K = len(self._symbol_spec)
        self._Pi  = numpy.zeros(self._M)       
                        
        # Map state names to indices
        for i in range(self._M):
            self._state_dic[ self._state_spec[i][0] ] = i
            self._state_rev[i] = self._state_spec[i][0]
            
        # Map symbol names to indices
        for i in range(self._K):
            self._symbol_dic[ self._symbol_spec[i] ] = i
            self._symbol_rev[i] = self._symbol_spec[i]
            
        # Extract transitions     
        self._Pij = self._extractTransitions()
        
        # Extract emissions 
        self._Pe = self._extractEmissions()
           
        # Extract start state
        self._Pi  = self._extractStartState()
        
        # Extract state_lead (states that can transition into state i)
        m = 0
        self._lead_state = [[] for i in range(self._M)]
        for j in range(self._M):
            for i in range(self._M):
                if self._Pij[i][j] > 0.0:
                    self._lead_state[j].append(i)
                    m += 1
        
        # Make a flat version of _lead information (for passing to viterbi)
        self._lead_vec = numpy.zeros(self._M + m)
        at = 0
        for j in range(self._M):
            self._lead_vec[at] = len(self._lead_state[j])
            at += 1
            for i in self._lead_state[j]: 
                self._lead_vec[at] = i
                at += 1

        
    def viterbi(self, seq = None, emissions = None):
        """ Calculate MLE viterbi on tokenized sequence or on generic emission probabilities """
        v = []
        if seq is not None:
            # Convert seq_lbl into symbol indices 
            seq_idx = [self._symbol_dic.get(s) for s in seq]
    
            # Call viterbi function in viterbi module
            # v = viterbi.viterbi(seq_idx, self._Pij, self._Pe, self._Pi)
            v = viterbi.viterbi(seq_idx, self._Pij, self._Pe, self._Pi, self._lead_vec)
            # v =  self.viterbi_slow(seq)
        elif emissions is not None and emissions.shape[1] == self._M:
            # Here emissions should be a matrix of shape (number of observations, M)
            seq_idx = range(0, emissions.shape[0])
            v = viterbi.viterbi(seq_idx, self._Pij, emissions, self._Pi, self._lead_vec)
            
        return v
    

    def _extractTransitions(self):
        """ Construct a matrix representation of the state transition probabilities. """
        # Extract transitions
        Pij = numpy.zeros((self._M, self._M))     
        for i in range(self._M):
            # Extract transitions 
            sum = 0
            # k_s = Transition state, v_s = transition weight
            for k_s, v_s in self._state_spec[i][1].iteritems():
                # Loop over states
                for k_j, v_j in self._state_dic.iteritems():
                    # Check if this state matches spec (only treat spec as a regex if it has a '*', otherwise prefixes match)
                    if re.search("\\*", k_s) and re.match(k_s, k_j):
                        # Wildcarded transition spec only match on or after current state
                        if v_j >= i:
                            # Decaying transition
                            p = v_s / float(1 + v_j - i)
                            Pij[i][v_j] += p 
                            sum += p
                    elif k_s == k_j:
                        # Non-wildcarded transition
                        Pij[i][v_j] += v_s
                        sum += v_s
                            
            if sum > 0:
                for j in range(self._M):
                    Pij[i][j] = Pij[i][j] / sum
        return Pij

    def _extractEmissions(self):
        """ Construct a matrix representation of the state emission probabilities, if present. """
        if self._state_spec[0][2] is not None:
            # Extract emissions 
            Pe  = numpy.zeros((self._M, self._K))    
            for i in range(self._M):               
                sum = 0               
                for k_s, v_s in self._state_spec[i][2].iteritems():
                    for k_k, v_k in self._symbol_dic.iteritems():
                        if (re.search("\\*", k_s) and re.match(k_s, k_k)) or k_s == k_k:
                            Pe[i][v_k] += v_s
                            sum += v_s
                if sum > 0:
                    for k in range(self._K):
                        Pe[i][k] = Pe[i][k] / sum
            return Pe

    def _extractStartState(self):
        """ Construct a matrix representation of the initial distribution. """
        # Extract start state
        Pi  = numpy.zeros(self._M)       
        sum = 0
        for k_s, v_s in self._start_spec.iteritems():
            for k, v in self._state_dic.iteritems():
                if (re.search("\\*", k_s) and re.match(k_s, k)) or k_s == k:
                    Pi[v] += v_s
                    sum += v_s
        if sum > 0:
            for i in range(self._M):
                Pi[i] = Pi[i] / sum
        else:
            # Default to uniform start
            for i in range(self._M):
                Pi[i] = 1.0 / self._M
        return Pi
    
    def encodeState(self, sta):
        """ Encode a state name to its state number (needed by users to map viterbi scores to states) """
        return Hmm.__dicLookup(self._state_dic, sta)
    
    def decodeState(self, sti):
        """ Decode a state number to its state name """
        return Hmm.__dicLookup(self._state_rev, sti)
    
    def encodeSymbol(self, sym):
        """ Encode a symbol name to its symbol number """
        return Hmm.__dicLookup(self._symbol_dic, sym)
    
    def decodeSymbol(self, syi):
        """ Encode a symbol number to its symbol name """
        return Hmm.__dicLookup(self._symbol_rev, syi)
       
    def print_Pij(self, out=sys.stdout):
        """ Print transition matrix """
        out.write("Pij = \n")
        for i in range(self._M):
            sum = 0
            out.write("[%3d,%20s] " % (i, self._state_rev[i][0:20]))
            for j in range(self._M):
                sum += self._Pij[i][j]
                out.write("%7.5f " % self._Pij[i][j])
                if j % 10 == 9:
                    out.write(" | ")
            out.write(" (sum = %7.5f)\n" % sum)
    
    def print_Pe(self, out=sys.stdout):
        """ Print emission matrix """
        out.write("Pe = \n")
        for i in range(self._M):
            sum = 0
            out.write("[%3d,%20s] " % (i, self._state_rev[i][0:20]))
            for j in range(self._K):
                sum += self._Pe[i][j]
                out.write("%7.5f " % self._Pe[i][j])
                if j % 10 == 9:
                    out.write(" | ")
            out.write(" (sum = %7.5f)\n" % sum)      
    
    def print_delta(self, delta, out=sys.stdout):
        """ Print dynamic programming matrix """
        out.write("delta = \n")
        N = numpy.shape(delta)[1] 
        for t in range(N):
            sum = 0
            out.write("[%3d,] " % t)
            for j in range(self._M):
                out.write("%4.3f " % delta[j][t])
                if j % 10 == 9:
                    out.write(" | ")
            out.write("\n")     

    def viterbi_slow(self, seq):
        """ Pure python version (slow) """
        pz = -1e300
        eps = 1e-300
        N = len(seq)
        delta = pz * numpy.ones((self._M, N))
        xsi = numpy.zeros((self._M, N), numpy.int8)
            
        # Convert seq_lbl into symbol indices (index '1' is the symbol name)
        seq_idx = [self._symbol_dic[s] for s in seq]
            
        # # Initialization
        for j in range(self._M):
            delta[j][0] = math.log(self._Pi[j] + eps) + math.log(self._Pe[j][seq_idx[0]] + eps)
                        
        # Recursion
        for t in range(1, N):
            for j in range(self._M):
                sup = pz 
                for i in self._lead_state[j]: 
                    term = delta[i][t-1] + math.log(self._Pij[i][j] + eps)
                    if sup < term:
                        sup = term
                        xsi[j][t] = i
                delta[j][t] = sup + math.log(self._Pe[j][seq_idx[t]] + eps)
                            
        # Termination (find max likelihood end point, append it to seq_tok tuple)
        vit = numpy.zeros(N, numpy.intc)
        dmax = delta[vit[N-1]][N-1]
        for i in range(1, self._M):
            if delta[i][N-1] > dmax:
                vit[N-1] = i
                dmax = delta[i][N-1]
                
        # Backtracking
        for t in range(N-2, -1, -1):
            vit[t] = xsi[vit[t+1]][t+1]
                
        return vit

    @staticmethod
    def __dicLookup(dic, key, unk=None):
        """ Lookup a value or a list of values in a dictionary """
        
        ans = []
        
        if not isinstance(key, list):
            key = [key]
            
        for k in key:
            if k in dic:
                # Known key 
                ans.append(dic[k])
            else:
                # Unknown state (lots of other ways to handle this)
                ans.append(unk)
            
        if len(key) == 1:
            # Convert answer back to a scalar if argument was a single string
            ans = ans[0]
            
        return ans

class TestHmm(unittest.TestCase):
    """ Test the hmm class """
    
    def setUp(self):
        """ Set model """
        start_spec = {}
        symbol_spec = ["uc", "lc", "ws"]
        state_spec = [("word",     {"word": 10, "space": 1}, {"uc":1, "lc":100}),
                      ("acronym",  {"acronym": 10, "space": 1}, {"uc":100}),
                      ("space",    {"word": 1, "acronym": 1, "space": 10}, {"ws":1})]
        
        self.hmm = Hmm(start_spec, symbol_spec, state_spec)

    def testCodeSymbol(self):
        """ Test decode/encode symbol """
        s1 = ["uc", "lc", "ws"]
        s2 = self.hmm.decodeSymbol(self.hmm.encodeSymbol(s1))
        self.assert_(s1 == s2)

    def testCodeState(self):
        """ Test decode/encode state """
        s1 = ["word", "acronym", "space"]
        s2 = self.hmm.decodeState(self.hmm.encodeState(s1))
        self.assert_(s1 == s2)
        
    def testViterbi(self):
        """ Test viterbi """
        v = self.hmm.viterbi("uc lc lc ws lc lc lc ws uc uc ws ws lc".split()) 
        self.assert_(all(v == numpy.array([0,0,0,2,0,0,0,2,1,1,2,2,0])))
    
""" Run test if run """
if __name__ == '__main__':
    unittest.main()
