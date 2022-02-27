#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os.path as op
import math
import os
import matplotlib
if 'DISPLAY' not in os.environ.keys():
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm
import matplotlib.colors
import seaborn as sns
from utils_nmf import eprint, validate_file
plt.rcParams.update({'font.size': 12})

# Plotting parameters:
NR_CHRS_XTICKS = 30         # number of characters to be printed of the xticks
FIG_SIZE = (15, 7)          # figure size
COLOR_MAP = 'tab10'         # color map. See https://matplotlib.org/users/colormaps.html
NCOL = 1

class PlotDeconv:
    def __init__(self, csv, min_rate=0.0, outpath=None, stubs=None,
                 full_legend=False):
        self.csv = csv
        self.min_rate = min_rate
        self.outpath = outpath
        self.full_legend = full_legend
        self.stubs = stubs
        self.validate_params()
        self.df = pd.read_csv(csv, index_col=0).fillna(0)
        self.plot_res()

    def validate_params(self):

        # input path
        validate_file(self.csv)

        # output path
        self.outpath = self.outpath
        if self.outpath is None:
            self.outpath = op.splitext(self.csv)[0] + '.pdf'
            self.outpath = op.splitext(self.csv)[0] + '.png'

        # min_rate
        min_rate = self.min_rate
        if not 100.0 >= min_rate >= 0:
            eprint('Min rate must be in range [0,100]')
            return
        if 1 > min_rate > 0:
            eprint('WARNING: min_rate is considered in percentages [0,100], not as a fraction [0,1]')
        self.min_rate = min_rate / 100.

    def hide_small_tissues(self):
        """
        tissues with very small contribution are grouped to the 'other' category.
        :return: The DataFrame with the new category ('other'),
                 where the low-contribution tissues are set to 0.
        """
        if self.min_rate <= 0:
            return
        others = self.df[self.df < self.min_rate].sum()
        self.df[self.df < self.min_rate] = 0.0
        self.df = self.df.append(others.rename('other'))

    def gen_bars_colors_hatches(self, nr_tissues):
        """
        Generate combinations of colors and hatches for the tissues bars
        Every tissue will get a tuple of (color, hatch)
        the last tuple is for the 'other' category, and is always black with no hatch.
        :return: a list of tuples, with length == nr_tissues
        """
        matplotlib.rcParams['hatch.linewidth'] = 0.3
        hatches = [None, 'xxx', '...', 'O', '++'][:max(1, nr_tissues // 7)]

        nr_colors = int(math.ceil(nr_tissues / len(hatches)) + 1)

        # generate bars colors:
        cmap = matplotlib.cm.get_cmap(COLOR_MAP)
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=float(nr_colors))
        colors = [cmap(norm(k)) for k in range(nr_colors)]

        def bar_tup_i(i):
            color_ind = i % nr_colors
            hatch_ind = int(i // math.ceil(nr_tissues / len(hatches)))
            return colors[color_ind], hatches[hatch_ind]

        colors_hatches_list = [bar_tup_i(i) for i in range(nr_tissues - 1)]
        return colors_hatches_list + [((0, 0, 0, 1), None)]

    def should_skip(self, i):
        if self.stubs is None:
            return False
        for s in self.stubs:
            if s.lower() in self.df.index[i].lower():
                return False
        return True

    def gray_out_absents(self, ch_list):
        for i in np.argwhere(self.df.values.max(axis=1) <= self.min_rate).flatten():
            ch_list[i] = ((0, 0, 0, 0), None)

    def plot_res(self):

        self.hide_small_tissues()
        nr_tissues, nr_samples = self.df.shape

        # generate bars colors and hatches:
        colors_hatches = self.gen_bars_colors_hatches(nr_tissues)
        if not self.full_legend:
            self.gray_out_absents(colors_hatches)

        plt.figure(figsize=FIG_SIZE)
        r = [i for i in range(nr_samples)]
        bottom = np.zeros(nr_samples)
        for i in range(nr_tissues):

            # skip ref cell types not in stubs
            if self.should_skip(i):
                continue

            plt.bar(r, list(self.df.iloc[i, :]),
                    edgecolor='white',
                    width=0.85,
                    label=self.df.index[i],
                    bottom=bottom,
                    color=colors_hatches[i][0],
                    hatch=colors_hatches[i][1])
            bottom += np.array(self.df.iloc[i, :])

        # Custom x axis
        plt.xticks(r, [w[:NR_CHRS_XTICKS] for w in self.df.columns], rotation='vertical', fontsize=12)
        yt = [0, .2, .4, .6, .8, 1.]
        ytl = [f'{int(x*100)}%' for x in yt]
        plt.yticks(yt, ytl, fontsize=12)
        plt.xlabel("sample")
        plt.xlim(-.6, nr_samples - .4)

        # Add a legend and a title
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=NCOL)
        plt.title('Deconvolution Results\n' + op.splitext(op.basename(self.outpath))[0])

        # adjust layout, save and show
        mm = 1 if NCOL == 1 else .83
        plt.tight_layout(rect=[0, 0, mm, 1])
        plt.savefig(self.outpath)
        eprint(f'Dumped figure to {self.outpath}')


def main():
    args = parse_args()

    # validate parameters
    PlotDeconv(csv=args.csv,
               min_rate=args.min_rate,
               outpath=args.outpath,
               stubs=args.stubs,
               full_legend=args.full_legend
               )
    if args.show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Deconvolution output csv to plot')
    parser.add_argument('--outpath', '-o', help='output. Default is the same name as CSV, but different suffix')
    parser.add_argument('--show', action='store_true', help='Show the figure in a pop up window')
    parser.add_argument('--min_rate', type=float, default=1.0,
            help='Cell types with smaller rates are combined to the "Other" category. Range 0-100 (percentages) [1]')
    parser.add_argument('--stubs', nargs='+',
            help='Show only reference cell types that match any of the stubs')
    parser.add_argument('--full_legend', action='store_true',
            help='For plotting: If set, legend show colors of all cell types' \
                    ' . Otherwise, show only for cell types found in at least one sample')
    return parser.parse_args()


if __name__ == '__main__':
    main()
