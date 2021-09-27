# Class for simulating spike trains based on different probability distributions

import numpy as np
import pandas as pd
import random

#import vonMises.vonMises as VM
#from process_trial_data import find_true_trial_end_samples


# def sim_binom_spktrn(probs):
#     """Simulate binary spike train from binomial probability distribution
#     Param:
#     probs: list of probabilities that unfold over time"""
#
#     binary_spktrn = []
#     for i in range(len(probs)):
#
#         binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#     return binary_spktrn

# def sim_binom_spktrn(probs,is_uniform,with_refractory,frate_target):
#     """Simulate binary spike train from binomial probability distribution
#     Param:
#     probs: list of probabilities that unfold over time
#     with_refractory: boolean, 1 to add refractory period
#     frate_target: scalar, firing rate target, in Hz"""
#     binary_spktrn = []
#     if frate_target == []:
#
#         if with_refractory == 0:
#
#             for i in range(len(probs)):
#
#                 binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#         else:
#
#             for i in range(len(probs)):
#
#                 tmp = np.random.binomial(1,probs[i])
#
#                 binary_spktrn.append(tmp)
#
#                 if (tmp == 1) & (i != len(probs)-1):
#                     probs[i+1] = 0.00001
#                     if (tmp == 1) & (i+1 != len(probs)-1):
#                         probs[i+2] = (0.00001)*.375 + 0.00001
#                         if (tmp == 1) & (i+2 != len(probs)-1):
#                             probs[i+3] = (0.00001)*.5 + 0.00001
#     else:
#
#         if is_uniform == 0:
#             rescale_factor = (frate_target/1000)*6
#             probs = probs*rescale_factor
#
#
#         if with_refractory == 0:
#
#             for i in range(len(probs)):
#
#                 binary_spktrn.append(np.random.binomial(1,probs[i]))
#
#         else:
#
#             for i in range(len(probs)):
#
#                 tmp = np.random.binomial(1,probs[i])
#
#                 binary_spktrn.append(tmp)
#
#                 if (tmp == 1) & (i != len(probs)-1):
#                     probs[i+1] = 0.00001
#                     if (tmp == 1) & (i+1 != len(probs)-1):
#                         probs[i+2] = (0.00001)*.375 + 0.00001
#                         if (tmp == 1) & (i+2 != len(probs)-1):
#                             probs[i+3] = (0.00001)*.5 + 0.00001
#
#
#     return binary_spktrn, probs
def sim_binom_spktrn(probs,is_uniform,with_refractory,frate_target):
    """Simulate binary spike train from binomial probability distribution
    Param:
    probs: list of probabilities that unfold over time
    with_refractory: boolean, 1 to add refractory period
    frate_target: scalar, firing rate target, in Hz"""
    binary_spktrn = []
    if frate_target == []:

        if with_refractory == 0:

            for i in range(len(probs)):

                binary_spktrn.append(np.random.binomial(1,probs[i]))

        else:

            for i in range(len(probs)):

                tmp = np.random.binomial(1,probs[i])

                binary_spktrn.append(tmp)

                if (tmp == 1) & (i != len(probs)-1):
                    probs[i+1] = 0.00001
                    if (tmp == 1) & (i+1 != len(probs)-1):
                        probs[i+2] = (0.00001)*.375 + 0.00001
                        if (tmp == 1) & (i+2 != len(probs)-1):
                            probs[i+3] = (0.00001)*.5 + 0.00001
    else:

        if is_uniform == 0:
            rescale_factor = (frate_target/1000)*6
            probs = probs*rescale_factor


        if with_refractory == 0:

            for i in range(len(probs)):

                binary_spktrn.append(np.random.binomial(1,probs[i]))

        else:

            for i in range(len(probs)):

                tmp = np.random.binomial(1,probs[i])

                binary_spktrn.append(tmp)

                if (tmp == 1) & (i != len(probs)-1):
                    probs[i+1] = 0.00001
                    if (tmp == 1) & (i+1 != len(probs)-1):
                        probs[i+2] = (0.00001)*.375 + 0.00001
                        if (tmp == 1) & (i+2 != len(probs)-1):
                            probs[i+3] = (0.00001)*.5 + 0.00001


    return binary_spktrn, probs


def make_phase_timeseries(freq,duration,fs):

    freq_mHz = freq/fs

    n_cycles_timeseries = freq_mHz * duration

    one_cycle = np.linspace(np.pi,-np.pi,int(1/freq_mHz))
    p = np.tile(one_cycle,int(n_cycles_timeseries))

    if len(p) < duration:
        p = np.hstack((p,one_cycle))
        p = p[:duration]

    phases = p

    return phases


# def find_true_trial_end_samples(n_trials,true_len_trial):
#     """Identifies and saves the time samples immediately following the last
#     time sample in a trial--immediately after so that you can slice with it
#     and know that you will select all samples up to but not including, the time
#     you have input"""
#
#     true_trial_ends_not_inclusive = []
#     for i in range(1,n_trials+1):
#         true_trial_ends_not_inclusive.append((true_len_trial-1)*i)
#
#     return true_trial_ends_not_inclusive


class Neuron:

    def __init__(self, avg_frate, n_trials, has_rhythm, has_refractory, freq_seq, mu, kappa, fs, len_trial):
        """freq_seq: list of frequency values that interneuron prefers and switches to within
        a single trial (e.g. interneuron goes from 8 Hz to 30 Hz to 8 Hz within single trial)
        has_refractory: boolean, 1: has a refractory period, 0: does not"""

        self.avg_frate = avg_frate
        self.n_trials = n_trials
        self.has_rhythm = has_rhythm
        self.has_refractory = has_refractory
        self.freq_seq = freq_seq
        self.mu = mu
        self.kappa = kappa
        self.fs = fs
        self.len_trial = len_trial

    def make_singletrial_phase(self):
        """mu: list of phase preferences for each frequency listed in freq_seq"""

        len_trial = self.len_trial
        freq_mHz = [i/self.fs for i in self.freq_seq]

        n_cycles_timeseries = [i * len_trial/len(self.freq_seq) for i in freq_mHz]

        p=[]
        mu_list=[]
        kappa_list=[]
        for i,val in enumerate(freq_mHz):
            tmp = np.linspace(np.pi,-np.pi,int(1/val))
            p.append(np.tile(tmp,int(n_cycles_timeseries[i])))

            mu_list.append(np.repeat(self.mu[i],len(p[i])))
            kappa_list.append(np.repeat(self.kappa[i],len(p[i])))

        self.phase_single_trial = [item for sublist in p for item in sublist]
        self.mus_single_trial = [item for sublist in mu_list for item in sublist]
        self.kappas_single_trial = [item for sublist in kappa_list for item in sublist]

        self.true_len_trial = len(self.phase_single_trial)

        return self.phase_single_trial, self.mus_single_trial,self.kappas_single_trial, self.true_len_trial

    def make_multitrial_phase(self):
        """"""

        d = {'phases': np.tile(self.phase_single_trial,self.n_trials),
            'mus': np.tile(self.mus_single_trial,self.n_trials),
            'kappas': np.tile(self.kappas_single_trial,self.n_trials)}

        self.df = pd.DataFrame(d)

        return self.df

    def make_spike_probs(self):
        """"""

        self.spike_probs=[]

        if self.has_rhythm == 1:
            for i,val in enumerate(self.df['mus'].values):

                p = VM.dvonmises([self.df['phases'].values[i]],
                                             val,
                                             self.df['kappas'].values[i])
                self.spike_probs.append(p[0])

        else:
            for i,val in enumerate(self.df['mus'].values):

                p = self.avg_frate/self.fs

                self.spike_probs.append(p)


        self.df['spike_probs'] = self.spike_probs # easier to work with this data type than just the spike_probs by self


        #sample spikes from binomial distribution
        if self.has_rhythm == 1:
            is_uniform = 0
        else:
            is_uniform = 1

        self.spikes, self.spike_probs = sim_binom_spktrn(self.df['spike_probs'].values,is_uniform,self.has_refractory,self.avg_frate)


        self.df['spike_probs'] = self.spike_probs
        self.df['spikes'] = self.spikes

        return self.df

    def label_trials(self):
        """"""

        true_trial_bins = find_true_trial_end_samples(self.n_trials,
                                                     self.len_trial)
        true_trial_bins.insert(0,0)
        labels = np.arange(len(true_trial_bins)-1)

        self.df['trial_labels'] = pd.cut(self.df.index.values,
                                         bins=true_trial_bins,
                                         labels=labels,
                                         include_lowest=True)
        return self.df

# class Neuron:
#
#     def __init__(self, avg_frate, n_trials, freq_seq, mu, kappa, fs, len_trial, has_refractory):
#         """freq_seq: list of frequency values that interneuron prefers and switches to within
#         a single trial (e.g. interneuron goes from 8 Hz to 30 Hz to 8 Hz within single trial)
#         has_refractory: boolean, 1: has a refractory period, 0: does not"""
#
#         self.avg_frate = avg_frate
#         self.n_trials = n_trials
#         self.freq_seq = freq_seq
#         self.mu = mu
#         self.kappa = kappa
#         self.fs = fs
#         self.len_trial = len_trial
#         self.has_refractory = has_refractory
#
#
#
#     def make_singletrial_phase(self):
#         """mu: list of phase preferences for each frequency listed in freq_seq"""
#
#         len_trial = self.len_trial
#         freq_mHz = [i/self.fs for i in self.freq_seq]
#
#         n_cycles_timeseries = [i * len_trial/len(self.freq_seq) for i in freq_mHz]
#
#         p=[]
#         mu_list=[]
#         kappa_list=[]
#         for i,val in enumerate(freq_mHz):
#             tmp = np.linspace(np.pi,-np.pi,int(1/val))
#             p.append(np.tile(tmp,int(n_cycles_timeseries[i])))
#
#             mu_list.append(np.repeat(self.mu[i],len(p[i])))
#             kappa_list.append(np.repeat(self.kappa[i],len(p[i])))
#
#         self.phase_single_trial = [item for sublist in p for item in sublist]
#         self.mus_single_trial = [item for sublist in mu_list for item in sublist]
#         self.kappas_single_trial = [item for sublist in kappa_list for item in sublist]
#
#         self.true_len_trial = len(self.phase_single_trial)
#
#         return self.phase_single_trial, self.mus_single_trial,self.kappas_single_trial, self.true_len_trial
#
#     def make_multitrial_phase(self):
#         """"""
#
#         d = {'phases': np.tile(self.phase_single_trial,self.n_trials),
#             'mus': np.tile(self.mus_single_trial,self.n_trials),
#             'kappas': np.tile(self.kappas_single_trial,self.n_trials)}
#
#         self.df = pd.DataFrame(d)
#
#         return self.df
#
#     def make_spike_probs(self):
#         """"""
#
#         self.spike_probs=[]
#
#         for i,val in enumerate(self.df['mus'].values):
#
#             p = VM.dvonmises([self.df['phases'].values[i]],
#                                          val,
#                                          self.df['kappas'].values[i])
#             self.spike_probs.append(p[0])
#
#         self.df['spike_probs'] = self.spike_probs # easier to work with this data type than just the spike_probs by self
#
#
#         #sample spikes from binomial distribution
#         self.spikes, self.spike_probs = sim_binom_spktrn(self.df['spike_probs'].values,self.has_refractory,self.avg_frate)
#
#
#         # self.spikes=[]
#         # for i in self.spike_probs:
#         #     self.spikes.append(np.random.binomial(1,i/binom_denom))
#
#         self.df['spike_probs'] = self.spike_probs
#         self.df['spikes'] = self.spikes
#
#         return self.df
#
#     def label_trials(self):
#         """"""
#
#         true_trial_bins = find_true_trial_end_samples(self.n_trials,
#                                                      self.len_trial)
#         true_trial_bins.insert(0,0)
#         labels = np.arange(len(true_trial_bins)-1)
#
#         self.df['trial_labels'] = pd.cut(self.df.index.values,
#                                          bins=true_trial_bins,
#                                          labels=labels,
#                                          include_lowest=True)
#         return self.df
