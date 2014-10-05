def test_band_pass_filter():
    sample_step = uerg.sec
    np.random.seed(13)
    white_noise = ContinuousDataEven((np.random.rand(2048) - 0.5)* uerg.mamp, sample_step)
    white_noise_spec = fft(white_noise)
    freq_range = Segment(np.array([0.3, 0.4]) * uerg.Hz)
    white_noise_filterred = band_pass_filter(white_noise, freq_range, 32)
    white_noise_filterred_spec = fft(white_noise_filterred)
    plot_quick(white_noise_spec, is_abs=True)
    plot_quick(white_noise_filterred_spec, is_abs=True)
    
    
    
    
# test_band_pass_filter()
    
    
#%%



    

