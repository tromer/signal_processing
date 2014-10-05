def test_read_wav():
    values = np.arange(10) * uerg.milliamp
    sample_rate = 1.0 * uerg.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)
    
    f_temp = tempfile.TemporaryFile()
    sp.io.wavfile.write(f_temp, sample_rate.magnitude, values.magnitude)
    sig_read = read_wav(f_temp)
    
    assert sig.is_close(sig_read)
    f_temp.close()
    
def test_write_wav():
    # copied from test_read_wav
    values = np.arange(10) * uerg.milliamp
    sample_rate = 1.0 * uerg.Hz
    sig = ContinuousDataEven(values, 1.0 / sample_rate)
    
    f_temp = tempfile.TemporaryFile()
    write_wav(sig, f_temp)
    sig_read = read_wav(f_temp)
    
    assert sig.is_close(sig_read)
    f_temp.close()    
    
test_read_wav()
test_write_wav()


