import h5py
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from convt3.parser import parse_lines, parse_lines_YDL, create_dataframe


class Session(object):
    def __init__(self, subj=None, sess=None, task=None):
        """
        A subject object that will have a time, cluster, subject, region, etc.
        and have methods for computing spatial metrics
        """

        self.subject = subj
        self.session = sess
        self.task = task
        self.path = f'/home1/salman.qasim/T3_Data/Subjects/{subj}/session_{sess}'
        self.events = None

    def _event_parser(self):
        """Parse the event structure from the logfile of the T3 task.
        Parameters
        ----------
        logfile_path : str
            Path to the logfile to parse.
        Notes
        -----
        Event structures are made at the sampling rate of the behavioral logile (~20 ms).
        One could imagine binning this for a lower sampling rate and consistent time bin sizes.
        These patients were run on a newer version of the task with a different
        logfile than the Baylor patients (See below)
        """

        # Parse the log file
        if self.subject == 'YDL':
            # This is a special case of a patient with an old logfile
            with open(self.log_path, 'r') as fobj:
                task = parse_lines_YDL(fobj)
        else:
            with open(self.log_path, 'r') as fobj:
                task = parse_lines(fobj)

        # Organize information into a dataframe
        self.events = create_dataframe(task)
