import argparse
import dotenv
import os
import src
import pathlib
import json
dotenv.load_dotenv(".env")

parser = argparse.ArgumentParser(description='Annotated corpus downloader.', usage='%(prog)s [options]')
parser.add_argument("--corpus", help="Subcorpus name [*|dental|!dental]", default="*")
parser.add_argument("--output_folder", help="Directory where the corpus is going to be downloaded.", default="annotated_corpus/")
parser.add_argument("--specialty_mapper", help="Specialty mapper json file.", default="specialty_mapper.json")
args = parser.parse_args()

documents = src.samples_loader_from_minio(
    os.environ.get('MINIO_SERVER'),
    os.environ.get('MINIO_ACCESS_KEY'),
    os.environ.get('MINIO_SECRET_KEY')
    )

with open(args.specialty_mapper) as j:
    specialty_mapper = json.load(j)

documents_folder = pathlib.Path(args.output_folder)

if not os.path.exists(documents_folder):
    os.makedirs(documents_folder)

if args.corpus == "*":
    documents_to_download = [document.name for document in documents]
elif args.corpus == "dental":
    documents_to_download = [name for name,specialty in specialty_mapper.items() if specialty in src.DENTAL_SPECIALTIES]
elif args.corpus == "not_dental":
    documents_to_download = [name for name,specialty in specialty_mapper.items() if not specialty in src.DENTAL_SPECIALTIES]
else:
    raise NotImplementedError

for document in documents:
    if document.name in documents_to_download:
        with open((documents_folder / document.name).with_suffix(".txt"),"w",encoding="utf-8") as t:
            t.write(document.text)
        with open((documents_folder / document.name).with_suffix(".ann"),"w",encoding="utf-8") as a:
            a.write(document.annotation)