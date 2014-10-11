import numpy as np


from signal_processing.extensions import plt_extension

def visual_test_plot_with_labels():
    x = np.arange(10)
    y = np.arange(10)
    x_label = "x"
    curv_label = "curv"
    plt_extension.plot_with_labels(x, y, x_label, curv_label)


"""
def visual_test_plot_quick():
    t = np.arange(10) * U_.sec
    vals = np.arange(10) * U_.volt
    sig = ContinuousData(vals, t)
    plot_quick(sig)

#visual_test_plot_quick()

def visual_test_plot():
    t = np.arange(10) * U_.sec
    vals = np.arange(10) * U_.volt
    sig = ContinuousData(vals, t)
    plot(sig)

# visual_test_plot_quick()

def visual_test_plot_under():
    t = np.arange(10) * U_.sec
    vals = np.arange(10) * U_.amp
    sig = ContinuousData(vals, t)
    sig_2 = sig
    sig_list = [sig, sig_2]
    plot_under(sig_list)

"""
