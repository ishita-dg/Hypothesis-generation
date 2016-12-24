import numpy as np
np.random.seed(3)

def knownQMR(D,S,C):
    """
    Known instantiation of disease-symptoms model
    
    diseases = {'other, 'lung cancer', 'TB', 'resp flu', 'cold',
                'gastroenteritis', 'stomach cancer', 'stomach flu', 'food poisoning'}
    symptoms = {'cough', 'fever', 'chest-pain', 'shortness-of-breath',
                 'nausea', 'fatigue','stomach cramps', 'abdominal pain'}
    """
    
    #Define prior probabilities
    # The "other" node is always on - to allow for a leak probability
    pis = np.array([1, 0.001, 0.05, 0.1, 0.2, 0.11, 0.05, 0.15, 0.2])
    
    # Define conditionals of sympotoms given only specific disease (mus)
    mus = np.array(
#{'other, 'lung cancer', 'TB', 'resp flu', 'cold', 'gastroenteritis', 'stomach cancer', 'stomach flu', 'food poisoning'}
        [[0.01, 0.3, 0.7, 0.05, 0.5, 0.0, 0.0, 0.0, 0.0], # Cough
         [0.01, 0.0, 0.1, 0.5, 0.3, 0.0, 0.0, 0.1, 0.2], # Fever
         [0.01, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Chest pain
         [0.01, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Shortness of breath
         [0.01, 0.0, 0.0, 0.2, 0.1, 0.5, 0.1, 0.5, 0.7], # nausea
         [0.01, 0.0, 0.0, 0.2, 0.3, 0.1, 0.05, 0.2, 0.4], # fatigue
         [0.01, 0.0, 0.0, 0.0, 0.0, 0.3, 0.05, 0.1, 0.5], # stomach cramps
         [0.01, 0.0, 0.0, 0.01, 0.0, 0.1, 0.5, 0.0, 0.0]  # abdominal pain
        ])
    
    cluster_assignments = np.array([0,0,0,0,1,1,1,1])
    symptom_assignments = np.array([0,0,0,0,1,1,1,1])
    
    return pis, mus, cluster_assignments, symptom_assignments
    

def inventQMR(D,S,C):
    """ 
    Returns random instantiation of disease-symptoms model
    """
    pis = np.random.beta(2,200,size = D)
    pis[0] = 1
    
    cluster_assignments = np.random.randint(0,C, size = D)
    symptom_assignments = np.random.randint(0,C, size = S)
    
    mus = np.zeros(shape = (S,D))
    
    # all intra cluster disease-symptom links
    for i in xrange(S):
        sa = symptom_assignments[i]
        mask = [cluster_assignments == sa]
        temp = np.random.uniform(size = D)
        mus[i,:] = mask * temp
        mus[i,0] = 0.01
    
    # Create some noise    
    
    # Select nonzero elements to turn to zero
    (x_nz, y_nz) = np.nonzero(mus[:,1:])
    nz = len(x_nz)
    nz_to_z = round(nz / 20)
    change_to_z = np.random.choice(np.arange(nz), 
                                   size = nz_to_z, replace = False)
    
    mus[x_nz[change_to_z], y_nz[change_to_z]+1] = 0

    # Select zeros to switch to nonzero
    (x_z, y_z) = np.where(mus[:,1:] == 0)
    zs = len(x_z)
    z_to_nz = round(zs / 20)
    change_to_nz = np.random.choice(np.arange(zs), 
                                    size = z_to_nz, replace = False)
    mus[x_z[change_to_nz], y_z[change_to_nz + 1]] = np.random.uniform(size = z_to_nz)
   
   
   # To make anchoring possible 
    min_symp = np.where(symptom_assignments == 
                        np.argmin(np.bincount(cluster_assignments)))[0]
    
    symp = np.random.choice(min_symp, size =1)
    c1 = np.argsort(np.bincount(cluster_assignments))[-1]
    c2 = np.argsort(np.bincount(cluster_assignments))[-2]
    
    d1 = np.random.choice(np.where(cluster_assignments == c1)[0], size = 1) + 1
    d2 = np.random.choice(np.where(cluster_assignments == c2)[0], size = 1) + 1
    
    mus[symp,d1] = 0.15
    mus[symp,d2] = 0.15
    cluster_assignments
    return pis, mus, cluster_assignments[1:], symptom_assignments