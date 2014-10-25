import os
from os import path
import shutil
import tempfile
import numpy as np
from signal_processing import U_, utils
from signal_processing.extensions import pint_extension
from signal_processing.segments.segments_obj import Segments
from signal_processing.segments import segments_of_continuous_obj
from signal_processing.segments.segments_of_continuous_obj import SegmentsOfContinuous
import signal_processing.continuous as cont
from signal_processing import ContinuousDataEven


def test_segments_of_continuous():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    segs = Segments([0, 5], [3, 6], U_.sec)
    x = SegmentsOfContinuous(segs, sig)

    assert x.segments.is_close(segs)
    assert x.source.is_close(sig)
    assert pint_extension.allclose(x.starts, segs.starts)
    assert pint_extension.allclose(x.ends, segs.ends)

    assert x.is_close(x)
    y = SegmentsOfContinuous(segs, sig * 2)
    assert not x.is_close(y)


def test_from_file():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    segs = Segments([0, 5], [3, 6], U_.sec)
    x = SegmentsOfContinuous(segs, sig)

    dir_temp = tempfile.mkdtemp()
    segs.to_csv(os.path.join(dir_temp, "segments.csv"))
    cont.io.write_wav(sig, os.path.join(dir_temp, "source.wav"))

    x_read = SegmentsOfContinuous.from_file(dir_temp)

    assert x.is_close(x_read)

    shutil.rmtree(dir_temp)


def test_to_file():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    segs = Segments([0, 5], [3, 6], U_.sec)
    x = SegmentsOfContinuous(segs, sig)

    dir_temp = tempfile.mkdtemp()
    x.to_file(dir_temp)
    x_read = SegmentsOfContinuous.from_file(dir_temp)

    assert x.is_close(x_read)

    shutil.rmtree(dir_temp)


def test_segs_from_dir():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    segs = Segments([0, 5], [3, 6], U_.sec)
    x = SegmentsOfContinuous(segs, sig)

    dir_temp = tempfile.mkdtemp()
    x.to_file(os.path.join(dir_temp, "1.segs"))
    x.to_file(os.path.join(dir_temp, "2.segs"))

    segs_list = segments_of_continuous_obj.segs_from_dir(dir_temp)
    expected_segs_list = [x, x]

    assert utils.is_close_many(segs_list, expected_segs_list)

    shutil.rmtree(dir_temp)

def test_segs_to_dir():
    sig = ContinuousDataEven(np.arange(10) * U_.mamp, U_.sec)
    segs = Segments([0, 5], [3, 6], U_.sec)
    x = SegmentsOfContinuous(segs, sig)
    segs_list = [x, x]

    dir_temp = tempfile.mkdtemp()
    segments_of_continuous_obj.segs_to_dir(segs_list, dir_temp)
    segs_list_read = segments_of_continuous_obj.segs_from_dir(dir_temp)

    assert utils.is_close_many(segs_list, segs_list_read)

    shutil.rmtree(dir_temp)
