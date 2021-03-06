
import warnings
import matplotlib.pyplot as plt
"""
TODO
-------
making pint work well with matplotlib
Herlpers: matplotlib.units?
http://matplotlib.org/examples/units/basic_units.html
make sure it works well also with share_x

refactor
----------------
try the object oriented interface to matplotlib.
if it works nicely, maybe it's better to migrate this package to work with the
object oriented interface

However, if we continue to work with the regular interface,
then maybe the functions shouldn't accept fig, subplot parametrs,
and the responsability for focusing would be of the caller
"""

# refactor
# -------------
# remove plot_quick function, it works with an old interface
#def plot_quick(contin, is_abs=False, fmt="-"):
    #"""
    #contin is a ContinuousData object
    #is_abs - whether to plot the abs of the y values. for spectrums.
    #Maybe - there sould be an input of lambda functions. to preprocess x,y
    #"""
    #raise NotImplementedError
    #warnings.warn("plot_quick is not tested")
    #TODO: test this function!
    # creat the figure here
    #return plot(contin, fig=None, is_abs=is_abs, fmt=fmt)
#%%


def focus_on_figure_and_subplot(fig, subplot, share_x=None, share_y=None):
    """
    focuses on required figure and subplot

    returns
    ----------
    fig : plt.figure object
    """
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    if subplot is not None:
        plt.subplot(*subplot, sharex=share_x, sharey=share_y)

    return fig


def plot_with_labels(x, y, x_label, curv_label,
                     fig=None, is_show_legend=True, subplot=None, share_x=None):
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
    warnings.warn("not tested")
    warnings.warn("plot dosn't rescale the last signal according to axes")

    # choose the appropriate figure and subplot
    fig = focus_on_figure_and_subplot(fig, subplot, share_x)

    # TODO add units to the label
    line = plt.plot(x, y, label=curv_label)[0]
    expand_y_lim(fig)
    expand_x_lim(fig)
    plt.xlabel(x_label)
    if is_show_legend:
        plt.legend(loc='best')

    return fig, line
    #raise NotImplementedError
        # return fig, axes??


def plot_few(*args):
    """
    plots few objects on the same figure

    TOOD
    ------
    add optional **kwargs and pass it to obj.plot()
    (for parameter of zoom, for instance)
    """
    fig =  args[0].plot()
    for obj in args[1:]:
        obj.plot(fig)

    return fig


def mark_vertical_lines(x_lines, fig, color='k', label=None):
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
    v_lines = plt.vlines(x_lines, y_min, y_max, colors=color, label=label)
    plt.legend(loc='best')
    return v_lines

def expand_y_lim(fig, ratio=0.2):
    plt.figure(fig.number)
    y_min, y_max = plt.ylim()
    dy = y_max - y_min
    y_min = y_min - 0.5 * ratio * dy
    y_max = y_max + 0.5 * ratio * dy
    plt.ylim(y_min, y_max)

def expand_x_lim(fig, ratio=0.1):
    plt.figure(fig.number)
    x_min, x_max = plt.xlim()
    dx = x_max - x_min
    x_min = x_min - 0.5 * ratio * dx
    x_max = x_max + 0.5 * ratio * dx
    plt.xlim(x_min, x_max)

def mark_horizontal_lines(y_lines, fig, label=None):
    """

    """
    warnings.warn("bad behaviour of units, just strips them, not tested")
    plt.figure(fig.number)
    x_min, x_max = plt.xlim()
    h_lines = plt.hlines(y_lines, x_min, x_max, label=label)
    plt.legend(loc='best')
    return h_lines


def plot_under(l,
               domain_range=None, is_show_legend=True, is_grid=True,
               y_lines=None):
    """
    plot a few signals one above the other




    OLD OLD OLD XXX
    add subplot of the signal, to an existing plot of another signal.
    the x axis would be coordinated.
    should enable easier examining of signals

    TODO: maybe add parameter of subplot or something
    """
    warnings.warn("not tested well for units, nots tested")
    fig = None

    if fig is not None:
        raise NotImplementedError

    fig = plt.figure()
    N = len(l)
    ax = plt.subplot(N, 1, 1)
    l[0].plot(fig, domain_range, is_show_legend)
    if is_grid:
        plt.grid()
    if y_lines is not None:
        mark_horizontal_lines(y_lines, fig)
    if is_show_legend is False:
        plt.legend().set_visible(False)

    for i in xrange(2, N + 1):
        focus_on_figure_and_subplot(fig, [N, 1, i], share_x=ax, share_y=ax)
        l[i - 1].plot(fig, domain_range, is_show_legend)

        if is_grid:
            plt.grid()
        if y_lines is not None:
            mark_horizontal_lines(y_lines, fig)
        if is_show_legend is False:
            plt.legend().set_visible(False)
        # lines.append(line)

    return fig
    #return f, lines
