
import os
import sys
import time
import pandas as pd
import os.path as op
import argparse
import subprocess
import multiprocessing
from multiprocessing import Pool
import hashlib
from pathlib import Path


dpath = str(Path(op.realpath(__file__)).parent)
DEF_NR_THREADS = multiprocessing.cpu_count()
SEP = ','

#################################
#                               #
#   General methods             #
#                               #
#################################

def eprint(*args,  **kargs):
    print(*args, file=sys.stderr, **kargs)


def validate_file(fpath):
    if not op.isfile(fpath):
        eprint('Invalid file', fpath)
        exit()
    return fpath


def check_executable(cmd):
    for p in os.environ['PATH'].split(":"):
        if os.access(op.join(p, cmd), os.X_OK):
            return
    eprint(f'executable {cmd} not found in PATH')
    exit(1)


def pat2name(pat):
    return op.basename(op.splitext(op.splitext(pat)[0])[0])


def drop_dup_keep_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def remove_files(files):
    for f in files:
        if op.isfile(f):
            os.remove(f)


def mkdir_p(dirpath):
    if not op.isdir(dirpath):
        os.mkdir(dirpath)
    return dirpath


def dump_df(fpath, df, verbose=True):
    df.to_csv(fpath, float_format='%.5f')
    if verbose:
        eprint(f'dumped {fpath}')

#################################
#                               #
#     NMF Specific logic        #
#                               #
#################################


def load_table(table_path, norm_cols=False):
    validate_file(table_path)
    df = pd.read_csv(table_path, sep=SEP, index_col=None)
    if df.shape[1] < 3:
        eprint(f'Invalid table: {table_path}. Too few columns ({df.shape[1]})')
        exit(1)
    df.columns = ['feature'] + list(df.columns)[1:]
    df.set_index('feature', inplace=True)
    if norm_cols:
        df = df.apply(lambda x: x / x.sum())
    return df


def parse_cut_str(cstr, maxlen):
    """
    Parse user input forr columns choice with unix cut syntax
    e.g. 1-4,5,19- will translate to choosings columns 1,2,3,4,5,19,...
    """
    if cstr is None:
        return []
    cstr = ''.join(cstr.split())  # remove whitespace
    if cstr.lower() == 'all' or cstr == '-':
        return list(range(maxlen))

    # validate string:
    for c in cstr:
        if c not in list(map(str, range(10))) + [',', '-']:
            eprint('Invalid input:', cstr)
            eprint('Only digits and [,-] characters are allowed')
            eprint(f'found: "{c}"')
            exit(1)

    # parse it
    include = []
    if cstr.endswith('-'):
        cstr += str(maxlen)
    elif cstr.startswith('-'):
        cstr = '1' + cstr
    for stub in cstr.split(','):
        if '-' not in stub:
            include.append(int(stub))
            continue
        start, end = map(int, stub.split('-'))
        assert end > start, 'range must be increasing!'
        include += list(range(start, min(end, maxlen) + 1))

    # another validation
    if len(set(include)) != len(include):
        eprint('Invalid input: duplicated columns', cstr)
        exit()
    # exclude = [x for x in range(1, maxlen + 1) if x not in include]
    # return include, exclude
    return sorted([x - 1 for x in include])

