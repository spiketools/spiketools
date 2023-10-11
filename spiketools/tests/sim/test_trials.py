"""Tests for spiketools.sim.trials"""

from spiketools.sim.trials import *

###################################################################################################
###################################################################################################

def test_sim_trials():

    n_trials = 2
    time_pre = 1
    time_post = 2

    for method, params in zip(['poisson'], [{'rate_pre' : 5, 'rate_post' : 10}]):

        trials = sim_trials(n_trials, method, time_pre, time_post, **params)

        assert isinstance(trials, list)
        assert len(trials) == n_trials
        for trial in trials:
            assert np.all(trial > -time_pre)
            assert np.all(trial < time_post)

def test_sim_trials_poisson():

    n_trials = 2
    rate_pre = 5
    rate_post = 10
    time_pre = 1
    time_post = 2

    trials = sim_trials_poisson(n_trials, rate_pre, rate_post, time_pre, time_post)
    assert isinstance(trials, list)
    assert len(trials) == n_trials
    for trial in trials:
        assert np.all(trial > -time_pre)
        assert np.all(trial < time_post)
