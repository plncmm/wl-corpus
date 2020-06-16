#!/usr/bin/env python3


import argparse
from src import Discard

parser = argparse.ArgumentParser(description='Discard diagnostic from txt file list.', usage='%(prog)s [options]')
parser.add_argument("--from_folder", help="Directory where the diagnostics are.", default="samples/")
parser.add_argument("--to_folder", help="Directory where the rejected diagnostics are.", default="samples_rejected/")
parser.add_argument("--file_list", help="Text file where the arguments to reject are, one filename per line.", required=True)

args = parser.parse_args()

d = Discard(args.from_folder,args.to_folder)
d.from_txt(args.file_list)
d.discard()