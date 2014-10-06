import warnings
import matplotlib.pyplot as plt

def mark_starts_ends(segments_, fig, color_start='r', color_end='g'):
    """
    get a figure and plot on it vertical lines according to
    starts and ends
    
    returns:
    -----------
    lines_start
    lines_end
    """
    warnings.warn("mark_starts_ends not tested, careful with units")
    plt.figure(fig.number)
    y_min, y_max = plt.ylim()
    starts_lines = plt.vlines(segments_.starts, y_min, y_max, colors=color_start, label='starts')
    ends_lines = plt.vlines(segments_.ends, y_min, y_max, colors=color_end, label='ends')
    plt.legend(loc='best')
    return starts_lines, ends_lines
    
def plot_quick(segments):
    raise NotImplementedError


