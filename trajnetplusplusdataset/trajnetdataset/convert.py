"""Create Trajnet data from original datasets."""
import argparse
import shutil

import pysparkling
import scipy.io

import readers
from scene import Scenes
from get_type import trajectory_type

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=9,
                        help='Length of observation')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Length of prediction')
    parser.add_argument('--train_fraction', default=0.6, type=float,
                        help='Training set fraction')
    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='Validation set fraction')
    parser.add_argument('--fps', default=2.5, type=float,
                        help='fps')
    parser.add_argument('--order_frames', action='store_true',
                        help='For CFF')
    parser.add_argument('--chunk_stride', type=int, default=2,
                        help='Sampling Stride')
    parser.add_argument('--min_length', default=0.0, type=float,
                        help='Min Length of Primary Trajectory')
    parser.add_argument('--synthetic', action='store_true',
                        help='convert synthetic datasets (if false, convert real)')
    parser.add_argument('--all_present', action='store_true',
                        help='filter scenes where all pedestrians present at all times')
    parser.add_argument('--goal_file', default=None,
                        help='Pkl file for goals (required for ORCA sensitive scene filtering)')
    parser.add_argument('--mode', default='default', choices=('default', 'trajnet'),
                        help='mode of ORCA scene generation (required for ORCA sensitive scene filtering)')

    # For Trajectory categorizing and filtering
    categorizers = parser.add_argument_group('categorizers')
    categorizers.add_argument('--static_threshold', type=float, default=1.0,
                              help='Type I static threshold')
    categorizers.add_argument('--linear_threshold', type=float, default=0.5,
                              help='Type II linear threshold (0.3 for Synthetic)')
    categorizers.add_argument('--inter_dist_thresh', type=float, default=5,
                              help='Type IIId distance threshold for cone')
    categorizers.add_argument('--inter_pos_range', type=float, default=15,
                              help='Type IIId angle threshold for cone (degrees)')
    categorizers.add_argument('--grp_dist_thresh', type=float, default=0.8,
                              help='Type IIIc distance threshold for group')
    categorizers.add_argument('--grp_std_thresh', type=float, default=0.2,
                              help='Type IIIc std deviation for group')
    categorizers.add_argument('--acceptance', nargs='+', type=float, default=[0.1, 1, 1, 1],
                              help='acceptance ratio of different trajectory (I, II, III, IV) types')

    args = parser.parse_args()
    sc = pysparkling.Context()

    # use our own labeling
    result = trajectory_type(track_id=0, args=args)


if __name__ == '__main__':
    main()
