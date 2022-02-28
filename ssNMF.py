#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

from utils_nmf import eprint, validate_file, load_table, \
    parse_cut_str, mkdir_p, dump_df
from algorithm import run_deconvolution
from parse_arguments import parse_args, validate_args


#############################################################
#                                                           #
#             Loading data                                  #
#                                                           #
#############################################################

def init_column(args, i, N):
    # TODO: use args for smarter initialization (e.g. beta distribution)
    vals = np.random.normal(loc=.75, scale=.5, size=N)
    return f'Unknown.{i + 1}', np.clip(vals, 0, 1)


def gen_NMF_atlas(args, features):
    df = pd.DataFrame(index=features)
    for i in range(args.nmf_cols):
        name, vals = init_column(args, i, len(features))
        df[name] = vals
    return df


def load_atlas(args, features):
    if args.atlas is None:
        return gen_NMF_atlas(args, features)

    # load atlas
    df = load_table(args.atlas)
    # append dummy columns:
    for i in range(args.add):
        name, vals = init_column(args, i, df.shape[0])
        if name in list(df.columns):
            eprint(f'Error: original input atlas contains a column named {name}')
            exit()
        df[name] = vals
    return df

#############################################################
#                                                           #
#             Parsing input                                 #
#                                                           #
#############################################################

def print_col_status(ocols, d):
    for tg in ('fixed', 'optimized', 'excluded', 'added'):
        lst = [(i, x) for i, x in enumerate(ocols) if i in d.keys() and d[i] == tg]
        eprint(f'{len(lst)} Columns are {tg}:')
        for i, f in lst:
            eprint(f'\t{i + 1}\t{f}')
        print()


def parse_cols(orig_atlas, args):
    """
    parse user input argument regrding the columns of
    the reference atlas:
        --optimize, --fix, --exclude, --add, --nmf_cols
    adjust the reference atlas accordingly:
        add/remove columns
        initialize empty columns
    return a binary vector bv where bv[i] == 1 iff column is fixed.
    """
    if args.nmf_cols:
        eprint(f'Full NMF mode. Init atlas with {args.nmf_cols} columns')
        return orig_atlas, np.zeros((args.nmf_cols,))

    n = orig_atlas.shape[1] - args.add

    # map index to status (excluded / fixed / optimized)
    def_mode = 'optimized' if args.fix else 'fixed'
    d = {i: def_mode for i in range(n)}
    for k, tg in zip((args.fix, args.optimize, args.exclude),
                     ('fixed', 'optimized', 'excluded')):
        # parse cut-like string
        for i in parse_cut_str(k, n):
            d[i] = tg
    # add additional (Unknown) columns:
    for i in range(n, orig_atlas.shape[1]):
        d[i] = 'added'

    # print status
    if args.verbose:
        print_col_status(orig_atlas.columns, d)

    # drop excluded columns from atlas
    atlas = orig_atlas.copy()
    todrop = [atlas.columns[i] for i in range(n) if d[i] == 'excluded']
    atlas.drop(todrop, axis=1, inplace=True)

    # build boolean status vector (i==1 iff column i is fixed)
    # map name to index
    rdn = {k: v for v, k in enumerate(orig_atlas.columns)}
    bv = np.where(pd.Series(atlas.columns).replace(rdn).\
            replace(d) == 'fixed', 1, 0)

    return atlas, np.array(bv)

#############################################################
#                                                           #
#             Main                                          #
#                                                           #
#############################################################


def main():
    args = parse_args()
    validate_args(args)

    if args.seed:
        np.random.seed(args.seed)

    # load samples table:
    sf = load_table(args.data, args.norm_data)
    features = sf.index.tolist()

    # load atlas:
    orig_atlas = load_atlas(args, features)
    atlas0, fixed_bv = parse_cols(orig_atlas, args)

    # make sure atlas and data have the exact same features (rows)
    assert (sf.index != atlas0.index).sum() == 0

    # deconvolve samples:
    A, Y, history = run_deconvolution(A         = atlas0.copy().values,
                                      X         = sf.copy().values,
                                      fixed     = fixed_bv,
                                      beta      = args.beta,
                                      eta       = args.eta,
                                      n_iter    = args.n_iter,
                                      normalize = not args.no_norm_weights)

    coef = pd.DataFrame(columns=sf.columns, index=atlas0.columns, data=Y)
    atlas = pd.DataFrame(columns=atlas0.columns, index=sf.index, data=A)

    # calc RMSE
    print(f'RMSE: {history[-1]}\n')

    # Dump results
    dump_df(args.prefix + '.atlas.csv', atlas)
    dump_df(args.prefix + '.coef.csv', coef)


if __name__ == '__main__':
    main()
