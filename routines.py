import numpy as np
import itertools as it
import qmr
reload(qmr)
import funcs
reload(funcs)
import tests as t
reload(t)
import demos as d
reload(d)

np.random.seed(36)

def RunTests(testMH_flag, scaling_flag, proposal, model):
    """
    Calls subroutines to run tests
    Gives values to variables requird for subroutines
    """
    if testMH_flag :
        """
        Define required parameters here
        """
        # positive symptoms
        pos = [1,4,5]
        # negative symptoms
        neg = [0,2]
        # rest assumed unknown
        N_steps = 30

        # Observation
        x = np.zeros(model.S)
        x[pos] = 1
        x[neg] = -1

        # fix dimensions
        #proposal.dict_fixdims.clear()
        #proposal.dict_fixdims[0] = 1

        t.testMH(x, N_steps, proposal, model)


    if scaling_flag :
        """
        Define required parameters here
        """
        # positive symptoms
        pos = [1,4,5]
        # negative symptoms
        neg = [0,2]
        # rest assumed unknown

        # Observation
        x = np.zeros(model.S)
        x[pos] = 1
        x[neg] = -1

        t.scaling(x, model)

    return



def RunDemos(proposal, model, c_a, s_a,
             variance_flag, subadd_flag, superadd_flag,
             anchoring_flag, conf_flag, dud_flag):
    """
    Determine the required varaiables in each demo in here
    c_a and s_a are the cluster and symptom assignments
    """

    if variance_flag :
        """
        Define required parameters here
        """
        check = proposal.clusters[1]
        possible = np.arange(1,8,1)
        print "Possible ds", possible
        pos = [1,4,5]
        x = np.zeros(model.S)
        x[pos] = 1
        true_p = model.exact_posterior(x)
        true = 0
        for i,mask in enumerate(it.product(*([[0,1]] * (model.D-1)))):
            z = np.array(mask)
            if np.sum(z[proposal.clusters[0]]) == 0:
                true += true_p[i]

        print true

        dist = model.prior()
        init_ps = dist[possible]
        init_ps /= sum(init_ps)

        n_runs = np.arange(1,24,1)
        N_steps = np.array([200])

        up = [0]

        d.variance(x, true, N_steps, n_runs, up, init_ps, possible, model, proposal, check)

    if (subadd_flag or superadd_flag) :
        """
        Define required parameters here
        """
        possible = proposal.clusters[1]
        print "Possible ds", possible
        pos = [1,4,5]
        #neg = [0,2,3,6,7]
        neg = []
        x = np.zeros(model.S)
        x[pos] = 1
        x[neg] = -1

        dist = model.prior()
        init_ps = dist[possible]
        init_ps /= sum(init_ps)

        #N_steps = np.arange(40, 3000, 300)
        N_steps = np.arange(20,110,10)

        if subadd_flag :
            #unpacked = w_o[:3]
            unpacked = [8,7]
            print "Subadd", unpacked
            n_runs = 100
            d.pvup(x, N_steps, n_runs, unpacked, init_ps, possible, model, proposal, 'sub')

        if superadd_flag :
            #unpacked = w_o[-3:]
            unpacked = [5,6]
            print "Superadd", unpacked
            n_runs = 40
            d.pvup(x, N_steps, n_runs, unpacked, init_ps, possible, model, proposal, 'super')


    if anchoring_flag :
        """
        Define required parameters here
        """
        start_c1 = [1]
        start_c2 = [6]
        c1 = proposal.which_cluster[start_c1[0]]
        c2 = proposal.which_cluster[start_c2[0]]
        pos = [1]
        #negative symptoms
        neg = []
        # rest assumed unknown

        # Observation
        x = np.zeros(model.S)
        x[pos] = 1
        x[neg] = -1

        n_runs = 400
        N_steps = np.arange(40, 500, 40)

        # fix dimensions
        proposal.update_fixdims({})

        d.anchoring(x, N_steps, n_runs, c1, c2, start_c1, start_c2,
                    model, proposal)

    if conf_flag :

        possible = proposal.clusters[0]
        pos = [1,5]
        x = np.zeros(model.S)
        x[pos] = 1

        dist = model.prior()
        init_ps = dist[possible]
        init_ps /= sum(init_ps)

        n_runs = 20
        #N_steps = np.arange(10, 250, 40)
        N_steps = np.arange(20,110,10)


        d.conf(x, N_steps, n_runs, init_ps, possible, model, proposal)


    if dud_flag :

        possible = proposal.clusters[1]
        print "Not possible ds", possible
        pos = [1,4,5]
        x = np.zeros(model.S)
        x[pos] = 1

        dist = model.prior()
        init_ps = dist[possible]
        init_ps /= sum(init_ps)


        n_runs = 100
        N_steps = np.arange(40, 3000, 300)


        unpacked = [5,6]
        print "Dud Alts", unpacked
        d.pvup(x, N_steps, n_runs, unpacked, init_ps, possible, model,
               proposal, 'dud')




    return
