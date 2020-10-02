#!/usr/bin/env python3

import argparse
from src import SamplePicker

parser = argparse.ArgumentParser(description='Diagnostic sample picker', usage='%(prog)s [options]')
parser.add_argument("--corpus_location", help="Location of the JSON corpus. [<json filename>|dw]", default="corpus.json")
parser.add_argument("--corpus", help="Subcorpus name [*|dental|!dental]", default="*")
parser.add_argument("--samples_folder", help="Directory where the diagnostics are.", default="samples/")
parser.add_argument("--samples_rejected_folder", help="Directory where the rejected diagnostics are.", default="samples_rejected/")
parser.add_argument("-n", help="number of diagnoses to pick.", required=True, type=int)

args = parser.parse_args()

if args.corpus_location == "dw":
    import sshtunnel
    import dotenv
    import os
    dotenv.load_dotenv(".env")
    server = sshtunnel.open_tunnel((os.environ.get("TUNNEL_HOST"), int(os.environ.get("TUNNEL_PORT"))),
            ssh_username=os.environ.get("TUNNEL_USER"),
            ssh_password=os.environ.get("TUNNEL_PASSWORD"),
            remote_bind_address=(os.environ.get("PG_HOST"), int(os.environ.get("PG_PORT"))))
    server.start()
    port = server.local_bind_port
else:
    port = None
p = SamplePicker(args.samples_folder,args.samples_rejected_folder,args.corpus_location,port,args.corpus)
p.pick(args.n)
if args.corpus_location == "dw":
    server.stop()