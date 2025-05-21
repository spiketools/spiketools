import numpy as np

def single_trial_occ(num_bins,min_occ,max_occ):
    return np.random.uniform(min_occ, max_occ, size=num_bins)  # Uniform occupancy in the range [0.5, 1.5] seconds


def single_uniform_occ(num_bins, scale):
    return np.ones(num_bins) * scale


def sim_occ_trials(num_trials, num_bins, min_occ, max_occ, use_random=False, uniform_scale=1):
    all_trials_occ = []

    for _ in range(num_trials):
        if use_random:
            trial_occ = np.random.uniform(min_occ, max_occ, size=num_bins)
        else:
            trial_occ = np.ones(num_bins) * uniform_scale
        all_trials_occ.append(trial_occ)

    all_trials_occ = np.array(all_trials_occ)
    avg_occ= np.sum(all_trials_occ, axis=0) / num_trials

    return all_trials_occ, avg_occ