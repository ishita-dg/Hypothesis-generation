import itertools as it
import operator
from operator import itemgetter 
import numpy as np
from scipy.stats import rv_discrete

#vpi = True
vpi = False

def discrete_samples(values, probabilities, size):
    """
    Draws "value" proportional to its "propbability", "size" times
    """
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

def sortdict (fixdims):
    """
    Takes fixed dimensions dict as input and outputs sorted array
    """
    sorted_fdims = np.array(sorted(fixdims.items(), key=operator.itemgetter(0)))
    return sorted_fdims

def array_to_str(arr):
    return ''.join(map(str, arr.astype(int)))
    
def update_states(current, new):
    """
    Collates new hypos with existing hypos
    """
    if current is None :
        return unique_states(new)
    else: return unique_states(np.vstack((current,new)))
    
def update_excl_prob(x, hypos, block,model):
    new_prob = 0
    if (block is not None):
        #mask = [np.sum(hypos[:,block], axis = 1) != 0]
        mask = [np.sum(hypos[:,block], axis = 1) == np.sum(hypos, axis = 1)-1]
        legit_hypos = hypos[mask]
        for h in legit_hypos : 
            if vpi : new_prob += np.exp(model.log_joint(x,h)) 
            else : new_prob += 1.0
        
    else:
        for h in hypos : 
            if vpi : new_prob += np.exp(model.log_joint(x,h)) 
            else : new_prob += 1.0
    
    return new_prob

def update_prob(x, hypos, block,model):
    new_prob = 0
    if (block is not None):
        mask = [np.sum(hypos[:,block], axis = 1) != 0]
        legit_hypos = hypos[mask]
        for h in legit_hypos : 
            if vpi : new_prob += np.exp(model.log_joint(x,h)) 
            else : new_prob += 1.0
        
    else:
        for h in hypos : 
            if vpi : new_prob += np.exp(model.log_joint(x,h)) 
            else : new_prob += 1.0
    
    return new_prob

def update_chainlength(i, N_steps, usedup):
    """
    Retrns chainlength according to whether or not in the first step
    """
    if i : return N_steps[i] - N_steps[i-1]
    else : return N_steps[i] - usedup

def unique_states(A):
    """
    Returns the unique hypos in the input array
    """
    a = A.copy()
    b = np.ascontiguousarray(a).view(np.dtype((np.void, 
                                               a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    
    unique_a = a[idx]
    
    if vpi : return unique_a
    else: return A

def setof_hypos(positives, D, sorted_fdims):
    """
    Takes as input the diseases to start from (postives)
    and returns an array of hypotheses vectors for each disease to start from
    """
    
    n_pos = len(positives)
    AllH = np.zeros((n_pos, D))
    AllH[:,sorted_fdims[:,0]] = sorted_fdims[:,1]
    AllH[np.arange(n_pos), positives] = 1    
    
    return AllH
                
