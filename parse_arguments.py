import argparse
from utils_nmf import eprint, mkdir_p


def validate_args(args):
    if args.nmf_cols:
        if args.add > 0:
            eprint('Invalid input: --add must be zero in NMF mode')
            exit()
        if args.optimize:
            eprint('Invalid input: --optimize in NMF mode ')
            exit()
        if args.fix:
            eprint('Invalid input: --fix in NMF mode ')
            exit()
        if args.nmf_cols < 2:
            eprint('Invalid input: --nmf_cols must be >=2 in NMF mode ')
            exit()
    if args.eta:
        assert args.eta >= 0, '--eta must be non negative'
    assert args.beta >= 0, '--beta must be non negative'
    assert args.n_iter > 0, '--n_iter must be positive'
    assert args.add >= 0, '--add must be non negative'

    pref = args.prefix
    if '/' in pref:
        mkdir_p(pref[:pref.rfind('/')])


def parse_args():
    parser = argparse.ArgumentParser()
    patlas = parser.add_mutually_exclusive_group(required=True)

    # required arguments
    patlas.add_argument('--atlas', '-a',
                        help='Path to the atlas (csv). If none specifed, '
                             'full NMF mode is assumed. It must have the '
                             'same index (first) column as the sample table'
                             ' (--data)')
    patlas.add_argument('--nmf_cols', '-m', type=int,
                        help='Number of columns for the atlas for the '
                             'NMF to learn')
    parser.add_argument('--data', '-i', required=True,
                        help='Sample/data table. A csv file')

    # arguments for the --atlas option:
    pcols = parser.add_mutually_exclusive_group()
    pcols.add_argument('--optimize',
                       help='Columns in the reference data to optimize. '
                            'Follows the same syntax as unix cut '
                            '(e.g. -2,4,6-7 will optimize columns 1,2,4,6,7). '
                            'The first column (feature/index) is not included '
                            'in this logic.')
    pcols.add_argument('--fix',
                       help='Columns in the reference data to fix. '
                            'Complementary to --optimize. '
                            'Follows the same syntax')
    parser.add_argument('--exclude',
                        help='Columns in the reference data to remove '
                             'completely from the atlas. Gets priorotized '
                             'over --fix and --optimize.')
    parser.add_argument('--add', type=int, default=0,
                        help='Number of additional ("unknown") columns '
                             'to append to the atlas. Default is 0. '
                             'Only works when --atlas is specifed, '
                             'not in NMF mode.')

    # Other
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of iterations for the (ss)NMF algorithm.')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='L1 regularization on proportions. '
                             'Must be non negative. '
                             'Default value is 0.0')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='L1 regularization on newly learned components. '
                             'Must be non negative. '
                             'Default value is 0.0')
    parser.add_argument('--norm_data', action='store_true',
                        help='normalize the input data such that each '
                             'sample will sum up to one.')
    parser.add_argument('--no_norm_weights', action='store_true',
                        help='Do not normalize the output weights. '
                             'If set, their sum may not sum up to one.')
    parser.add_argument('--prefix', '-p', default='./out',
                        help='prefix for output files (csv and png)')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--plot', action='store_true')
    # parser.add_argument('--threads', '-@', type=int, default=DEF_NR_THREADS,
    #                     help='Number of threads [cpu_count()]')
    return parser.parse_args()


