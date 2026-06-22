"""Default settings for plots."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

###################################################################################################
###################################################################################################

# Define a list of arguments that will be caught and applied with 'set'
SET_KWARGS = ['title', 'xlim', 'ylim', 'xlabel', 'ylabel',
              'xticks', 'yticks', 'xticklabels', 'yticklabels']

# Define a list of other arguments to be caught
OTHER_KWARGS = ['legend']

# Define list of default plot colors, making sure colors are hex, for downstream consistency
#   Hex encoding changed in mpl: https://github.com/matplotlib/matplotlib/issues/29915
DEFAULT_COLORS = [to_hex(col) for col in plt.rcParams['axes.prop_cycle'].by_key()['color']] \
    if plt else None

# Define default settings for plotting text
TEXT_SETTINGS = {'fontdict' : {'fontsize' : 14}, 'ha' : 'center', 'va' : 'center'}
