#!/usr/bin/python3 -u

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

def print_col_status(ocols, d, n):
    def prep_v(tg):
        lst = [(x, ocols[x]) for x in range(n) if d[x] == tg]
        eprint(f'{len(lst)} Columns are {tg}:')
        for i, f in lst:
            eprint(f'\t{i + 1}\t{f}')
    prep_v('fixed')
    prep_v('optimized')
    prep_v('excluded')


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

    # parse cut-like strings:
    op_i = parse_cut_str(args.optimize, n)
    fix_i = parse_cut_str(args.fix, n)
    ex_i = parse_cut_str(args.exclude, n)

    # map index to status (excluded / fixed / optimized)
    d = {i: None for i in range(n)}
    for i in ex_i:
        d[i] = 'excluded'
    for i in fix_i:
        if d[i] != 'excluded':
            d[i] = 'fixed'
    for i in op_i:
        if d[i] != 'excluded':
            d[i] = 'optimized'
    def_mode = 'optimized' if args.fix else 'fixed'
    for i in range(n):
        if d[i] is None:
            d[i] = def_mode

    # print status
    if args.verbose:
        print_col_status(orig_atlas.columns, d, n)

    # drop excluded columns from atlas
    atlas = orig_atlas.copy()
    todrop = [atlas.columns[i] for i in range(n) if d[i] == 'excluded']
    atlas.drop(todrop, axis=1, inplace=True)

    # make boolean status vector (i==1 iff column i is fixed)
    # map name to index
    rdn = {k: v for v, k in enumerate(orig_atlas.columns)}
    bv = np.zeros((atlas.shape[1], ))
    for i in range(atlas.shape[1]):
        if 'Unknown.' in atlas.columns[i]:
            continue
        if d[rdn[atlas.columns[i]]] == 'fixed':
            bv[i] = 1

    return atlas, np.array(bv)

#############################################################
#                                                           #
#             Main                                          #
#                                                           #
#############################################################


def main():
    args = parse_args()

    validate_args(args)

    # load samples table:
    sf = load_table(args.data)
    sample_names = list(sf.columns)
    features = sf.index.tolist()

    # load atlas:
    orig_atlas = load_atlas(args, features)
    orig_atlas, fixed = parse_cols(orig_atlas, args)
    ref_samples = list(orig_atlas.columns)

    # make sure atlas and data have the exact same features (rows)
    assert (sf.index != orig_atlas.index).sum() == 0

    # deconvolve samples:
    A, Y, history = run_deconvolution(A=orig_atlas.copy().values,
                                      X=sf.copy().values,
                                      fixed=fixed,
                                      beta=args.beta,
                                      eta=args.eta,
                                      n_iter=args.n_iter)

    coef = pd.DataFrame(columns=sample_names, index=ref_samples, data=Y)
    atlas = pd.DataFrame(columns=ref_samples, index=features, data=A)
    print('A', atlas.shape)
    print('Y', coef.shape)

    # calc RMSE
    print('RMSE:', history[-1])


    # Dump results
    orig_atlas_path = args.prefix + '.atlas.orig.csv'
    atlas_path = args.prefix + '.atlas.csv'
    coef_path = args.prefix + '.coef.csv'
    dump_df(atlas_path, atlas)
    dump_df(orig_atlas_path, orig_atlas)
    dump_df(coef_path, coef)

    # plot results
    if args.plot:
        if args.verbose:
            eprint('Plotting results')
        plot_results(args, orig_atlas_path, atlas_path, coef_path)


def plot_results(args, orig_atlas_path, atlas_path, coef_path):

    from plot_atlas import plot_atlas, plot_weights
    from plot_deconv import PlotDeconv
    plot_atlas(atlas_path)
    plot_atlas(orig_atlas_path, nan_orig=True)
    # plot_weights(coef_path)
    PlotDeconv(csv=coef_path)


if __name__ == '__main__':
    main()
