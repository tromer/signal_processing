import os
import glob


def read_dir(path, suffix, reader, params=[]):
    """
    reads all the files of given suffix, from the dir

    params
    ---------
    path : str
        directory

    suffix : str
        wav, txt, csv etc.

    reader : function

    params : list
        more parameters the reader

    returns
    ----------
    out_list : list
        list of objects, that the reader returns

    """
    if not os.path.isdir(path):
        raise ValueError("path should be a dir")

    out_list = []
    print "directory: , ", path
    print "path:, ", os.path.join(path, "*." + suffix)
    files = glob.glob(os.path.join(path, "*." + suffix))
    print files
    for f in files:
        curr_object = reader(f, *params)
        out_list.append(curr_object)

    return out_list

def write_dir(l, path, suffix, writer):
    """
    writes an objects list to directory
    the names are serial at the moment

    params
    ----------
    l : list

    path : str
        directory

    suffix : str
        wav, txt etc.

    writer : function

    returns
    -----------
    None

    """
    if not os.path.isdir(path):
        raise ValueError("path should be a dir")
    for i in xrange(len(l)):
        curr_object = l[i]
        name = str(i)
        writer(curr_object, os.path.join(path, '.'.join([name, suffix])))

