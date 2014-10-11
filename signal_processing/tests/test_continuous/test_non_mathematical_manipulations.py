import numpy as np

from signal_processing import U_
from signal_processing.continuous.continuous_data_even_obj import ContinuousDataEven

from signal_processing.continuous.non_mathematical_manipulations import concatenate

def test_concatenate():
    sig_1 = ContinuousDataEven(np.arange(32) * U_.mamp, U_.sec)
    chunks = sig_1.get_chunks(15 * U_.sec)
    """
    print len(chunks)
    print sig_1.values
    print chunks[0].values
    """
    sig_2 = concatenate(chunks)
    assert sig_1.is_close(sig_2)



