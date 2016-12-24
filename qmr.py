import numpy as np
import itertools as it
from scipy.misc import logsumexp
import funcs
reload(funcs)

## TODO : convert dict fixdims to array -1 for unfixed, 0/1 for value fixed to
    
class QMRModel():
    """
    Simple disease -> symptom model
    """
    def __init__(self, D, S, nus, thetas):
        """
        :param D:   Num diseases
        :param S:   Num symptoms
        :param nus: log prior over diseases
        :param thetas:   S x D matrix of -log(1-mus) where mus are weights
        """
        self.D, self.S, self.nus, self.thetas = D, S, nus, thetas
        self.control = False

    
    def log_joint(self, x, z):
        """
        Returns logjoint posterior probability P(x,z) (unnormalised)
        """
        #assert z.dtype == np.bool
        xpos = x.copy()
        xneg = x.copy()
        
        xpos[xpos == -1] = 0 
        xneg[xneg == 1] = 0 
        xneg = -xneg
        
        lp = 0
        lp += np.dot(self.nus, z)
        lp += np.dot(np.log(1- np.exp(-np.dot(self.thetas,z))), xpos)
        lp += -np.dot(np.dot(self.thetas,z), xneg)
        lp += 2
        return lp

    def exact_posterior(self, x):
        """
        :param x:   S vector of observed symptoms 1 for there, 
                    0 for not known/there?
        :return:    Matrix of probabilities for each of the 2^D disease combos
        """
        S, D, nus, thetas = self.S, self.D, self.nus, self.thetas
        
        logpost = np.zeros(2**(D-1))
        for i,mask in enumerate(it.product(*([[0,1]] * (D-1)))):
            
            z = np.array(mask)
            z = np.insert(z,0,1)
            logpost[i] = self.log_joint(x, z)

        # Normalize the posterior
        post = np.exp(logpost - logsumexp(logpost))
        assert np.allclose(post.sum(), 1.0)

        return post

    def sample(self):
        """
        Sample the generative model
        """
        S, D, nus, thetas = self.S, self.D, self.nus, self.thetas
        z = np.random.rand(D) < np.exp(nus)
        z = np.nonzero(z)[0] 
        
        x = np.random.rand(S) < 1-np.exp(- np.sum(thetas[:,z], axis = 1))

        return z, x

    def MHchain(self, x, start, transition_proposal, n_steps):
        """
        Run a MHChain.
        
        Keyword arguments : 
        x -- symptoms vector
        start -- initial disease vector
        transition_proposal -- function that returns a new state, 
                                given input state
        
        Returns : 
        States visited (including start state, excluding last state) 
        as array of vectors
        
        Last state as a vector
        
        Variable "accepts" calculates 
        the number of shifts to new states the chain made (not returned)
        
        """
        
        zs = np.zeros((n_steps + 1, self.D))
        zs[0] = start
        accepts = []

        if (self.control):
            count = 1
            ids = np.random.choice(np.arange(0,256), size = n_steps, replace = False)
            for i,mask in enumerate(it.product(*([[0,1]] * (self.D-1)))):                
                temp = np.array(mask)
                temp = np.insert(temp,0,1)
                if (i in ids): 
                    zs[count] = temp
                    count += 1
                    del temp
                    
            return zs[:-1], zs[-1]
        
        for i in xrange(n_steps):
            current = zs[i]
            current_lp = self.log_joint(x, current)
            
            z_prop = transition_proposal(current)
            
            prop_lp = self.log_joint(x, z_prop)
            
            # acceptance rate calculation
            log_alpha = prop_lp - current_lp
            alpha = np.exp(log_alpha)
            u = np.random.uniform()
            
            if (u) < alpha:
                # Accept
                zs[i+1] = z_prop
                accepts.append(i)
            else:
                # Stay put
                zs[i+1] = zs[i]
                
        return zs[:-1], zs[-1]
    
    def prior(self):
        return np.exp(self.nus)
    


class ProposalBuilder():
    """
    A class to make a proposal mechanism for decision tree based
    proposals
    """
    def __init__(self, model, inertia_flag, simi_flag, dd_flag, p_inertia, p_simi,
                 which_cluster, clusters):
        """
        :param model : QMR model
        :param inertia_flag : Implement exponential decay with 
                            increase in diseases y/n
        :param simi_flag : Implement preferentially proposng similar
                            diseases y/n
        :param dd_flag : Implement datadriven initialiser y/n
        :param p_inertia : Probability of sticking to current sparsity
        :param p_simi : Probability of ticking to current cluster
        :param which_cluster : Dict, returns which cluster input disease is in
        :param clusters : Dict, returns diseases in input cluster
        :param dict_fixdims : Default value - dict with leak node fixed 1
        :param sorted_fixdims : Sorted array version of dict
        """
        
        self.model, self.inertia_flag = model, inertia_flag 
        self.simi_flag, self.dd_flag = simi_flag, dd_flag
        self.p_inertia, self.p_simi = p_inertia, p_simi
        self.which_cluster, self.clusters = which_cluster, clusters
        
        # Only fix leak node to 1
        self.dict_fixdims = {0:1}
        # sort fixed dimensions when chain initialised
        self.sorted_fixdims = funcs.sortdict(self.dict_fixdims)
        
    def update_fixdims(self,update):
        """
        Takes as inupt a dictionary and updates current dictionary
        """        
        self.dict_fixdims.clear()
        self.dict_fixdims[0] = 1
        self.dict_fixdims.update(update)
        self.sorted_fixdims = funcs.sortdict(self.dict_fixdims)        
        
        return
    
    def initialise(self, x):
        """
        Takes as input the current state and returns 
        the starting point of chain
        Initialises the sorted array of fixed dimensions, state with fixed Dims
        """
        
        state = np.zeros(self.model.D)
        
        for disease in self.dict_fixdims:
            state[disease] = self.dict_fixdims[disease]
        
        dd_prop  = self.TabularBayes(x)
        n_prop = self.model.prior()
        
        if self.inertia_flag :
            if self.dd_flag : return self.SingleSample(dd_prop, state)
            else: return self.SingleSample(n_prop, state)
        else :
            if self.dd_flag : return self.IndependentSample(dd_prop, state)
            else: return self.IndependentSample(n_prop, state)
            
    def IndependentSample(self, dist, state):
        """
        Returns a state with each unfixed disease samples independently from 
        the input distribution
        """
        sample_from = dist.copy()
        new_state = state.copy()
        
        sampleable = [x for x in np.arange(self.model.D) 
                      if x not in self.sorted_fixdims]
        
        for disease in sampleable:
            if (np.random.rand(1)[0] < sample_from[disease]):
                new_state[disease] = 1        
        
        return new_state
    
    def SingleSample(self, dist, state):
        """
        Returns a state with one of the unfixed diseases sampled from the 
        input distribution
        """
        
        sample_from = dist.copy()
        new_state = state.copy()
        
        sampleable = [x for x in np.arange(self.model.D) 
                      if x not in self.sorted_fixdims]
        
        # normalise
        sample_dist = sample_from[sampleable]
        sample_dist /= np.sum(sample_dist)
        
        sample = funcs.discrete_samples(sampleable,sample_dist,1)
        
        new_state[sample] = 1        
        
        return new_state
        
    def transition(self, state):
        """
        Input : current state,
        fixdims are an array of sorted fixed dimensions
        Output : (proposed) next state
        """
        if self.inertia_flag :
            # Activate preference for inertia vectors
            if (np.random.rand(1)[0] > self.p_inertia):
                new_state = self.randomswitch(state)
            else :
                new_state = self.resample(state)
        else:
            if self.simi_flag:
                new_state = self.simiswitch(state)
            else:
                new_state = self.randomswitch(state)
            
        return new_state
    
    def randomswitch(self, state): 
        
        """
        Returns new state with all states randomly chosen
        """
        new_state = np.random.binomial(1, 0.5, size=self.model.D)
        new_state[0] = 1
        #changeable = [x for x in np.arange(state.size) 
                      #if x not in self.sorted_fixdims]
        
        #switchpoint = np.random.choice(changeable)
        #new_state[switchpoint] = not new_state[switchpoint]        
        
        return new_state

    def simiswitch(self, state): 
        """
        Returns new state with a 0 or 1 randomly switched
        """
        diseases = np.nonzero(state)[0]
        ids = []
        for d in diseases:
            if d != 0:
                ids.append(self.which_cluster[d])
        
        ids = np.array(ids)
        
        if len(ids) != 0:
            dom = np.argmax(np.bincount(ids))
        else :
            dom = np.random.choice(np.arange(len(self.clusters)))
        
        new_state = np.ones(self.model.D)
        
        for i in xrange(self.model.D - 1):
            i = i+1
            if self.which_cluster[i] == dom :
                p = self.p_simi
            else:
                p = 1.0 - self.p_simi
            new_state[i] = np.random.binomial(1, p, size= 1)
        
        return new_state
      
    def resample(self, state):
        """
        Returns new state with same number of, but altered diseases
        """
        
        new_state = state.copy()
        
        diseases = np.nonzero(state)
        changeable_from = [x for x in diseases[0] 
                           if x not in self.sorted_fixdims]
        
        if len(changeable_from) == 0:
            # In this case, no switchable diseases so randomswitch
            return self.randomswitch(state)
        
        old_disease = np.random.choice(changeable_from)
        
        if self.simi_flag and np.random.rand(1)[0] < self.p_simi :
            
            possible_diseases = self.clusters[self.which_cluster[old_disease]]
            changeable_to = [x for x in possible_diseases
                             if x not in self.sorted_fixdims]
            new_disease = np.random.choice(changeable_to)
        else:
            changeable_to = [x for x in np.arange(self.model.D) 
                             if x not in self.sorted_fixdims]
            new_disease = np.random.choice(changeable_to)
            
        new_state[old_disease] = 0 
        new_state[new_disease] = 1 
        
        return new_state
    
    def TabularBayes(self, x):
        """
        Takes as input a a set of symptoms 
        and returns a distribution over diseases
        As if only one disease caused the symptoms
        """
        logpost = np.zeros(self.model.D)
        states = np.eye(self.model.D, dtype=int)
        for i in xrange(self.model.D):
            logpost[i] = self.model.log_joint(x,states[i])
        
        return np.exp(logpost)
        
    
    
