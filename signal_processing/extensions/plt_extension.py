
import warnings
import matplotlib.pyplot as plt
from signal_processing import uerg


def plot_quick(contin, is_abs=False, fmt="-"):
    """
    contin is a ContinuousData object
    is_abs - whether to plot the abs of the y values. for spectrums.
    Maybe - there sould be an input of lambda functions. to preprocess x,y
    """
    raise NotImplementedError
    warnings.warn("plot_quick is not tested")
    #TODO: test this function!
    # creat the figure here
    return plot(contin, fig=None, is_abs=is_abs, fmt=fmt)
#%%
    
def plot_with_labels(x, y, x_label, curv_label, fig=None, subplot=None, share_x=None):
    """
    Note
    -----------
    a refactoring happened here. documentation may me old

    add a plot of ContinuousData instance, to an existing figure
    TODO: allow passing every parameter the plt.plot accepts. i.e - making ot a complete
    wrapper around plt.plot
    TODO: make sure somehow that all the plots on the same figure, share x axis dimensionality
    and rescale them - (fig, x_untis) tuple
    TODO: instead of putting units on y axis, use legend and put units there
    
    parameters:
    -------------
    contin
    fig - a plt.plot figure object
    subplot - list indicating subplot like [3,1,1] - of 3 lines, and 1 col, subplot 1 (on top)
    x_label : str
    
    curv_label : str
    """
    # assert contin type?
    # TODO: add support for legend
    warnings.warn("plot is not tested")
    warnings.warn("plot dosn't rescale the last signal according to axes")
   
    # choose the appropriate figure and subplot
    if fig == None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
        
    if subplot != None:
        plt.subplot(*subplot, sharex=share_x)
            
    # TODO add units to the label
    line = plt.plot(x, y, label=curv_label)[0]
    plt.xlabel(x_label) 
    plt.legend(loc='best')

    return fig, line 
    #raise NotImplementedError
        # return fig, axes??


def plot_few(*args):
    """
    plots few objects on the same figure
    
    """
    fig =  args[0].plot()
    for obj in args[1:]:
        obj.plot(fig)

    return fig


def mark_vertical_lines(x_lines, fig, label=None):
    """
    get a figure and plot on it vertical lines according to
    starts and ends
    
    returns:
    -----------
    lines_start
    lines_end
    """
    # this is the signature of the oroginal function, that this function
    # was refactored from.
    # def mark_starts_ends(segments_, fig, color_start='r', color_end='g'):
    warnings.warn("not tested, careful with units")
    plt.figure(fig.number)
    y_min, y_max = plt.ylim()
    v_lines = plt.vlines(x_lines, y_min, y_max, label=label)
    plt.legend(loc='best')
    return v_lines

def mark_horizontal_lines(y_lines, fig, label=None):
    """
    
    """
    warnings.warn("bad behaviour of units, just strips them, not tested")
    plt.figure(fig.number)
    x_min, x_max = plt.xlim()
    h_lines = plt.hlines(y_lines, x_min, x_max, label=label)
    plt.legend(loc='best')
    return h_lines
 
def plot_under(contin_list, fig=None, is_abs=False, fmt="-"):
    """
    plot a few signals one above the other
    
    
    
    
    OLD OLD OLD XXX
    add subplot of the signal, to an existing plot of another signal.
    the x axis would be coordinated.
    should enable easier examining of signals
    
    TODO: maybe add parameter of subplot or something
    """
    raise NotImplementedError
    warnings.warn("not tested well for units, nots tested")
    if fig != None:
        raise NotImplementedError
    
    f = plt.figure()
    lines = []
    N = len(contin_list)
    ax = plt.subplot(N, 1, 1)

    for i in xrange(N):
        junk, line = plot(contin_list[i], f, [N, 1, i + 1], share_x=ax)
        lines.append(line)
    
    return f, lines
    

