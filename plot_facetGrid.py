#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os.path as op
import os
import matplotlib
if 'DISPLAY' not in os.environ.keys():
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
from utils_nmf import eprint, validate_file

# Plotting parameters:
NCOL = 6
plt.rcParams.update({'font.size': 12})


def load_data(args):
    validate_file(args.csv)
    df = pd.read_csv(args.csv)
    df.columns = ['component'] + list(df.columns)[1:]
    if args.include:
        df = df[df['component'].isin(args.include)]
    elif args.exclude:
        df = df[~df['component'].isin(args.exclude)]
    return df


def main():
    args = parse_args()

    df = load_data(args)
    dm = df.melt(id_vars='component', var_name='samples', value_name='rate')
    targets = [x for x in df['component'].tolist() if 'Unknown' in x]
    dm['target'] = dm['component'].isin(targets)

    yticks = np.arange(0, 1.001, .2)
    ylabs = [f'{int(x*100)}%' for x in yticks]

    g = sns.FacetGrid(dm, col="component", col_wrap=args.col_wrap, hue='target')
    g.map_dataframe(sns.barplot, x="samples", y="rate")
    g.set(ylim=(0, 1.0) , yticks=yticks,  yticklabels=ylabs, xticks=[])
    g.set_titles(col_template="{col_name}")

    if args.outpath:
        plt.savefig(args.outpath)

    if args.show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('csv', help='Deconvolution output csv to plot')
    oparser = parser.add_mutually_exclusive_group(required=True)
    oparser.add_argument('--outpath', '-o', help='output. Default is the same name as CSV, but different suffix')
    oparser.add_argument('--show', action='store_true', help='Show the figure in a pop up window')

    # optional arguments
    parser.add_argument('--col_wrap', type=int, default=NCOL,
            help=f'Number of columns of output [{NCOL}]')
    cparser = parser.add_mutually_exclusive_group()
    cparser.add_argument('--include', nargs='+',
            help='Show only components that match any of these')
    cparser.add_argument('--exclude', nargs='+',
            help='Drop these components')
    return parser.parse_args()


if __name__ == '__main__':
    main()
