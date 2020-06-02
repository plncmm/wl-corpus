#!/usr/bin/env python3

import argparse
from src import SamplePicker

parser = argparse.ArgumentParser(description='Diagnostic sample picker', usage='%(prog)s [options]')
parser.add_argument("--corpus", help="Location of the JSON corpus.", default="corpus.json")
parser.add_argument("--samples_folder", help="Directory where the diagnostics are.", default="samples/")
parser.add_argument("--samples_rejected_folder", help="Directory where the rejected diagnostics are.", default="samples_rejected/")
parser.add_argument("-n", help="number of diagnoses to pick.", required=True, type=int)

args = parser.parse_args()

p = SamplePicker(args.corpus,args.samples_folder,args.samples_rejected_folder)
p.pick(args.n)