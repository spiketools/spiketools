"""Default settings for plots."""

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

# Define a list of arguments that will be caught and applied with 'set'
SET_KWARGS = ['title', 'xlim', 'ylim', 'xlabel', 'ylabel',
              'xticks', 'yticks', 'xticklabels', 'yticklabels']

# Define a list of other arguments to be caught
OTHER_KWARGS = ['legend']

# Collect a list of the default matplotlib color cycle
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define default settings for plotting text
TEXT_SETTINGS = {'fontdict' : {'fontsize' : 14}, 'ha' : 'center', 'va' : 'center'}
