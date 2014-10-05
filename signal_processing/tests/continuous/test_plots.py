def visual_test_plot_quick():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    plot_quick(sig)
    
#visual_test_plot_quick()

def visual_test_plot():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.volt
    sig = ContinuousData(vals, t)
    plot(sig)
    
# visual_test_plot_quick()

def visual_test_plot_under():
    t = np.arange(10) * uerg.sec
    vals = np.arange(10) * uerg.amp
    sig = ContinuousData(vals, t)
    sig_2 = sig
    sig_list = [sig, sig_2]
    plot_under(sig_list)


