import numpy as np
import matplotlib.pyplot as plt
import funcs
import itertools as it
reload(funcs)
import qmr
reload(qmr)

def variance(x, true, N_steps, n_runs, up, init_ps, possible, model, proposal, check):
    """
    Plots the variance of the net
    generated probability, with chain length
    """
    l = 7

    N_steps = np.arange(1,l+1,1)*N_steps[0]

    ps = packed(x, N_steps, n_runs[-1], up, init_ps, possible, model, proposal, check, all_ps = 1,refresh = True)

    response1_all = np.array(ps[0])

    avgs = np.arange(1,l+1,1)
    betw_error = np.zeros(l)
    self_error = np.zeros(l)

    for i in avgs:
        temp_betw = []
        temp_self = []
        for comb in it.combinations(np.arange(n_runs[-1]), i):
            indices = np.array(comb)
            temp_betw.append(np.mean(response1_all[indices]))

        temp_self = np.mean(np.array(ps[:i]), axis = 0)
        betw_error[i-1] = np.mean((temp_betw - true)**2)#np.std(temp)
        self_error[i-1] = np.mean((temp_self - true)**2)

    #same_error = np.mean(((response1_all + est1)/2 - true)**2) #np.std((response1_all + est1)/2)
    #same_error0 = np.mean(((response1_all + est2)/2 - true)**2) #np.std((response1_all + est2)/2)


    # Plotting
    variance = plt.figure()
    ax = variance.add_subplot(111)
    ax.plot(avgs,np.zeros(avgs.size), linewidth = 2.0, c='g')
    ax.plot(avgs,betw_error, linewidth = 2.0, c='b', label = 'Avg across subject')
    ax.scatter(avgs,betw_error, marker='o', c='b', s=30)

    ax.plot(avgs,self_error, linewidth = 2.0, c='r', label = 'Avg within subject')
    ax.scatter(avgs,self_error, marker='o', c='r', s=30)

    #ax.scatter(2, same_error, marker='o', c='r', s=30, label = 'Avg within subject close')


    ax.set_title('Standard Error in QMR Markov Chain model', fontsize = 22)
    ax.set_ylabel('Standard Error of probability estimate', fontsize = 20)
    ax.set_xlabel('Number of chains/subjects averaged over', fontsize = 20)
    ax.legend(fontsize = 18)
    ax.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()

    np.savetxt('data/variance.txt', (avgs, betw_error, self_error),
                   fmt = ('%0.4f'),  delimiter = ' ',
                   header = 'N_chains, average across subjects')

    fh = file('data/variance.txt','a')
    #np.savetxt(fh, (same_error, same_error0), fmt = ('%0.4f'),  delimiter = ' ',
               #header = 'average within subjects (close), average within subject (far)')
    fh.close()



    return


def packed(x, N_steps, n_runs, up, init_ps, possible,
           model, proposal, check = None, hypos = None, all_ps = None, refresh = False):

    #np.random.seed(132)
    if check is None :
        check = possible
    apac = np.zeros(shape = (len(N_steps), n_runs))
    possible_hypos = funcs.setof_hypos(possible, model.D,
                                     proposal.sorted_fixdims)

    for j in xrange(n_runs):
        # Averaging over trials

        p_gen_hypos = None
        initial_state_p = possible_hypos[funcs.discrete_samples(
            np.arange(len(init_ps)), np.array(init_ps), len(up))]
        #print "packed inits", initial_state_p

        for i in xrange(len(N_steps)):
            # Probabilities at increasing chain length
            # Determine the length of chain run at each iteration
            n_steps = np.ones(len(up), dtype=int)
            total_steps = funcs.update_chainlength(i, N_steps, 0)
            n_steps *= int(total_steps/len(up))
            inc = np.random.randint(len(up))
            n_steps[inc] += total_steps - sum(n_steps)
            ## stochastic generation of p hypotheses
            if refresh : p_gen_hypos = None

            for i_p in xrange(len(up)):
                new_hypos, last_state = model.MHchain(x, initial_state_p[i_p],
                                                           proposal.transition, n_steps[i_p])

                initial_state_p[i_p] = last_state.copy()
                p_gen_hypos = funcs.update_states(p_gen_hypos, new_hypos)
                del new_hypos, last_state

            p_prob = funcs.update_excl_prob(x, p_gen_hypos, None, model)
            p_hypo_prob = funcs.update_excl_prob(x, p_gen_hypos, check, model)

            apac[i][j] = p_hypo_prob/p_prob
            #print "Packed", N_steps[i], p_prob, p_hypo_prob

    if hypos is not None:
        return p_gen_hypos
    elif all_ps is not None:
        return apac
    else:
        return np.mean(apac, axis = 1), np.std(apac, axis = 1)

def unpacked(x, N_steps, n_runs, up, init_ps, possible,
             model, proposal, check = None, hypos = None):

    if check is None :
            check = possible
    aupac = np.zeros(shape = (len(N_steps), n_runs))
    up_hypos = funcs.setof_hypos(up, model.D,
                                     proposal.sorted_fixdims)

    for j in xrange(n_runs):
        # Averaging over trials
        up_gen_hypos = None
        initial_state_up = list(up_hypos)

        for i in xrange(len(N_steps)):
            # Probabilities at increasing chain length

            # Determine the length of chain run at each iteration
            n_steps = np.ones(len(up), dtype=int)
            total_steps = funcs.update_chainlength(i, N_steps, 0)
            n_steps *= int(total_steps/len(up))
            inc = np.random.randint(len(up))
            n_steps[inc] += total_steps - sum(n_steps)

            ##stochastic generation of up hypotheses

            for i_up in xrange(len(up)):
                new_hypos, last_state = model.MHchain(x, initial_state_up[i_up],
                                                           proposal.transition,
                                                           n_steps[i_up])
                initial_state_up[i_up] = last_state.copy()
                up_gen_hypos = funcs.update_states(up_gen_hypos, new_hypos)
                del new_hypos, last_state

            up_prob = funcs.update_prob(x, up_gen_hypos, None, model)
            up_hypo_prob = funcs.update_excl_prob(x, up_gen_hypos, check, model)

            aupac[i][j] = up_hypo_prob/up_prob
            #print "Unpacked", N_steps[i], up_prob, up_hypo_prob


    if hypos is not None:
        return p_gen_hypos
    else :
        return np.mean(aupac, axis = 1), np.std(aupac, axis = 1)


def pvup(x, N_steps, n_runs, up, init_ps, possible, model, proposal, which= None) :

    print x, up
    """
    Plots sub or super additivity of probability judgments
    """
    pac, a = packed(x, N_steps, n_runs, up, init_ps, possible, model, proposal)
    upac, a = unpacked(x, N_steps, n_runs, up, init_ps, possible, model, proposal)
    diff = upac - pac

    # Plotting
    if (which is not None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(N_steps,np.zeros(N_steps.size), linewidth = 2.0, c='g',
                label = 'Packed')
        ax.set_ylabel('Difference between unpacked and packed', fontsize = 20)
        ax.tick_params(axis='both', labelsize=15)

    else:
        print "Returned"
        return diff[:-1]


    if (which == 'sub'):
        ax.plot(N_steps[:-1],diff[:-1], linewidth = 2.0, c='r',
                    label = 'Typical unpacked')
        ax.scatter(N_steps[:-1],diff[:-1], marker='o', c='r', s=30)
        ax.set_title('Subadditivity in Markov Chain model', y = 1.04, fontsize = 22)
        ax.set_xlabel('Number of Samples (analog for time)', fontsize = 20)
        ax.legend(fontsize = 18)
        plt.show()

        np.savetxt('data/subadditivity.txt', (N_steps[:-1], diff[:-1]),
                           fmt = ('%0.4f'),  delimiter = ' ',
                           header = 'xvals,yvals')


    elif (which == 'super'):
        ax.plot(N_steps[:-1],diff[:-1], linewidth = 2.0, c='b',
                    label = 'Atypical unpacked')
        ax.scatter(N_steps[:-1],diff[:-1], marker='o', c='b', s=30)
        ax.set_title('Superadditivity in Markov Chain model', y = 1.04, fontsize = 22)
        ax.set_xlabel('Length of chain (analog for time)', fontsize = 20)
        ax.legend(fontsize = 18)
        ax.set_ylim(-0.35, 0.1)
        plt.show()
        np.savetxt('data/superadditivity.txt', (N_steps[:-1], diff[:-1]),
                   fmt = ('%0.4f'),  delimiter = ' ',
                   header = 'xvals,yvals')

        np.savetxt('data/weakevi0.txt', (N_steps[:-1], upac[:-1], pac[:-1]),
                           fmt = ('%0.4f'),  delimiter = ' ',
                           header = 'xvals,yvals')


    elif (which == 'dud'):
            ax.plot(N_steps[:-1],-diff[:-1], linewidth = 2.0, c='b',
                        label = 'With dud')
            ax.scatter(N_steps[:-1],-diff[:-1], marker='o', c='b', s=30)
            ax.set_title('Dud Alternative effect in Markov Chain model', y = 1.04, fontsize = 22)
            ax.set_xlabel('Length of chain (analog for time)', fontsize = 20)
            ax.legend(fontsize = 18)
            ax.set_ylim(-0.1, 0.35)
            plt.show()
            np.savetxt('data/dudalt.txt', (N_steps[:-1], -diff[:-1]),
                       fmt = ('%0.4f'),  delimiter = ' ',
                       header = 'xvals,yvals')

    return

def anchoring(x, N_steps, n_runs, c1, c2, start_c1, start_c2, model, proposal):

    c1start = funcs.setof_hypos(start_c1, model.D, proposal.sorted_fixdims)
    c2start = funcs.setof_hypos(start_c2, model.D, proposal.sorted_fixdims)
    c1states = proposal.clusters[c1]
    c2states = proposal.clusters[c2]

    diff_c1chain = np.zeros(len(N_steps))
    diff_c2chain = np.zeros(len(N_steps))
    diff_net = np.zeros(len(N_steps))

    for j in xrange(n_runs):
        # Averaging over trials

        gen_hypos = None
        initial_state_c1chain = c1start.copy()
        initial_state_c2chain = c2start.copy()
        gen_hypos_c1chain = None
        gen_hypos_c2chain = None

        for i in xrange(len(N_steps)):

            foo = 0

            # Determine the length of chain run at each iteration
            n_steps = funcs.update_chainlength(i, N_steps, 0)

            ## stochastic generation of hypotheses starting in c1
            new_hypos_c1, last_state = model.MHchain(x, initial_state_c1chain,
                                                  proposal.transition,
                                                  n_steps)
            initial_state_c1chain = last_state.copy()
            gen_hypos_c1chain = funcs.update_states(gen_hypos_c1chain,
                                                    new_hypos_c1)

            ## stochastic generation of hypotheses starting in c2
            new_hypos_c2, last_state = model.MHchain(x, initial_state_c2chain,
                                                  proposal.transition,
                                                  n_steps)
            initial_state_c2chain = last_state.copy()
            gen_hypos_c2chain = funcs.update_states(gen_hypos_c2chain,
                                                    new_hypos_c2)


            norm_c1chain = funcs.update_prob(x, gen_hypos_c1chain, None, model)
            norm_c2chain = funcs.update_prob(x, gen_hypos_c2chain, None, model)

            #c1states_c1chain = (funcs.update_excl_prob(x, gen_hypos_c1chain, [3], model)
                                #/norm_c1chain)
            #c2states_c1chain = (funcs.update_excl_prob(x, gen_hypos_c1chain, [7], model)
                                #/norm_c1chain)
            #c1states_c2chain = (funcs.update_excl_prob(x, gen_hypos_c2chain, [3], model)
                                #/norm_c2chain)
            #c2states_c2chain = (funcs.update_excl_prob(x, gen_hypos_c2chain, [7], model)
                                #/norm_c2chain)

            c1states_c1chain = (funcs.update_prob(x, gen_hypos_c1chain, [3], model)
                                /norm_c1chain)
            c2states_c1chain = (funcs.update_prob(x, gen_hypos_c1chain, [7], model)
                                /norm_c1chain)
            c1states_c2chain = (funcs.update_prob(x, gen_hypos_c2chain, [3], model)
                                /norm_c2chain)
            c2states_c2chain = (funcs.update_prob(x, gen_hypos_c2chain, [7], model)
                                /norm_c2chain)
            #print "c1 chain", c1states_c1chain, c2states_c1chain
            #print "c2 chain", c1states_c2chain, c2states_c2chain
            # Take difference
            diff_c1chain[i] += c1states_c1chain - c2states_c1chain
            diff_c2chain[i] += c1states_c2chain - c2states_c2chain
            diff_net[i] = diff_c1chain[i] - diff_c2chain[i]

    diff_c1chain /= n_runs
    diff_c2chain /= n_runs
    diff_net /= n_runs

    # Plotting
    anchoring = plt.figure()
    ax = anchoring.add_subplot(111)

    ax.scatter(N_steps,diff_net, marker='o', c='k', s=30)
    ax.plot(N_steps,diff_net, linewidth = 2.0, c='k',
            label = 'Net diff')

    # ax.plot(N_steps,diff_c1chain, linewidth = 2.0, c='r',
                # label = 'Starting from cluster 1', ls = '--')
    #ax.scatter(N_steps,diff_c1chain, marker='o', c='r', s=30)
    # ax.plot(N_steps,diff_c2chain, linewidth = 2.0, c='b',
            # label = 'Starting from cluster 2', ls = '--')
    #ax.scatter(N_steps,diff_c2chain, marker='o', c='b', s=30)

    ax.plot(N_steps,np.zeros(N_steps.size), linewidth = 2.0, c='g')
    ax.set_title('Anchoring in QMR Markov Chain model', y = 1.04, fontsize = 22)
    ax.set_ylabel('Difference in Probabilities', fontsize = 20)
    ax.set_xlabel('Length of chain (analog for time)', fontsize = 20)
    ax.tick_params(axis='both', labelsize=15)
    #ax.set_ylim(-0.5, 0.8)
    ax.legend(fontsize = 18)
    plt.show()

    np.savetxt('data/anchoring.txt', (N_steps, diff_net, diff_c1chain, diff_c2chain),
               fmt = ('%0.4f'),  delimiter = ' ',
               header = 'xvals,black, red, blue')



    return

def conf(x, N_steps, n_runs, init_ps, possible, model, proposal):


    # Displays the top 5 diseases from exact

    edist = model.exact_posterior(x)
    #exact_ps = dist[possible]
    #exact_ps /= sum(exact_ps)
    exact_ps = edist
    order = np.argsort(-exact_ps)
    care = order[:5]
    print "possibilities", possible
    print "EXACT:"
    for i,mask in enumerate(it.product(*([[0,1]] * (model.D-1)))):
        z = np.array(mask)
        z = np.insert(z,0,1)
        if i in care:
            print np.where(care == i)[0][0], np.nonzero(z)[0][1:]


    pac_h = packed(x, [256], n_runs, [1], init_ps, possible,
                   model, proposal, check = None, hypos = 1)

    gen_ps = np.zeros(len(possible))
    for i in xrange(len(possible)):
        gen_ps[i] = funcs.update_excl_prob(x, pac_h, [possible[i]], model)

    order = np.argsort(-gen_ps)
    print "GENERATED"
    print possible[order]
    print gen_ps[order]

    gd = possible[order[:2]]
    print gd

    gen, foo = packed(x, N_steps, n_runs, gd, init_ps, possible,
                        model, proposal, check = gd, hypos = None)
    given, foo = unpacked(x, N_steps, n_runs, gd, init_ps, possible,
                            model, proposal, check = gd, hypos = None)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.scatter(N_steps,gen, marker='o', c='r', s=30)
    #ax.plot(N_steps,gen, linewidth = 2.0, c='r',
            #label = 'Self Generated')
    #ax.scatter(N_steps,given, marker='o', c='b', s=30)
    #ax.plot(N_steps,given, linewidth = 2.0, c='b',
            #label = 'Presented')
    ax.scatter(N_steps,given - gen, marker='o', c='r', s=30)
    ax.plot(N_steps,given-gen, linewidth = 2.0, c='r',
            label = 'Difference')

    ax.plot(N_steps,np.zeros(N_steps.size), linewidth = 2.0, c='g')
    ax.set_title('Confidence in QMR Markov Chain model', y = 1.04, fontsize = 22)
    ax.set_ylabel('Probabilities for Given - Self Generated', fontsize = 20)
    ax.set_xlabel('Length of chain (analog for time)', fontsize = 20)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize = 18)
    plt.show()

    np.savetxt('data/confidence.txt', (N_steps, given-gen), fmt = ('%0.4f'),
                   delimiter = ' ', header = 'xvals,yvals')
