import numpy as np

import qmr
reload(qmr)
import funcs
reload(funcs)
import routines
reload(routines)
import dataf
reload(dataf)

## Defining the model

#D = 6  # Num of diseases (including leak node)
#S = 20  # Num of symptoms
#C = 2  # Num of clusters
#pis, mus, cluster_assignments, symptom_assignments = data.inventQMR(D,S,C)

D, S, C = 9, 8, 2
pis, mus, cluster_assignments, symptom_assignments = dataf.knownQMR(D,S,C)

nus = np.log(pis)
thetas = -np.log(1-mus)

model = qmr.QMRModel(D, S, nus, thetas)

## Defining the proposal distribution

clusters = {}
for i in xrange(C):
        clusters[i] = np.where(cluster_assignments == i)[0] + 1

which_cluster = {}
for i in xrange(D-1):
        which_cluster[i+1] = cluster_assignments[i]

# The options for initialising are with prior, datadriven with Tabular Bayes
dd_flag = 0

# Probability of proposing hypothesis decays with reduction in sparsity
inertia_flag = 0
# Probability that we stick to current sparsity level
p_inertia = 1.0

# Probability of proposing hypothesis closer is higher
simi_flag = 1
# Probability that we stick to the current cluster
p_simi = 0.7

proposal = qmr.ProposalBuilder(model,inertia_flag, simi_flag, dd_flag,
                               p_inertia, p_simi, which_cluster, clusters)

# All tests
testMH_flag = 1
scaling_flag = 0

#routines.RunTests(testMH_flag, scaling_flag, proposal, model)

model.control = False
# All demos

flags = [0,0,0,1,1,1]
variance_flag = flags[0]
subadd_flag = flags[1]
superadd_flag = flags[2]
anchoring_flag = flags[3]
conf_flag = flags[4]
dud_flag = flags[5]

routines.RunDemos(proposal, model, cluster_assignments, symptom_assignments,
                  variance_flag, subadd_flag, superadd_flag,
                  anchoring_flag, conf_flag, dud_flag)
