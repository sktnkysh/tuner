#!/usr/bin/env python

import os
import sys
import argparse
from tuner import load_data

parser = argparse.ArgumentParser()
parser.add_argument('src_dir', nargs='?')
#parser.add_argument('src_dir', type=str)
parser.add_argument('-o', '--out', dest='dst_dir', default=None, type=str)
parser.add_argument('--brain', action='store_true')
parser.add_argument('-t', '--teaching-file', dest='teaching_file', default=None, type=str)
args = parser.parse_args()

if args.src_dir:
    src_dir = args.src_dir
elif not sys.stdin.isatty():
    src_dir = sys.stdin.read().rstrip()
else:
    parser.print_help()

dst_dir =\
    args.dst_dir if args.dst_dir is not None else\
    os.path.join('classed_datasets', src_dir.split('/')[-1])

if args.teaching_file is not None:
    df = load_data.df_fromdir_eyes(src_dir, args.teaching_file)
elif args.brain:
    df = load_data.df_fromdir_brain(src_dir)
else:
    print('Specify --teaching-file or --brain')
    sys.exit()

load_data.classed_dir_fromdf(df, dst_dir)
print(dst_dir)
