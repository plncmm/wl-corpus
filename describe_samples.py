import src
import dotenv
import os
import logging
import pandas as pd
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.pipeline
import json
import numpy as np
dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

port = src.create_tunnel(
    os.environ.get("TUNNEL_HOST"),
    os.environ.get("TUNNEL_PORT"),
    os.environ.get("TUNNEL_USER"),
    os.environ.get("TUNNEL_PASSWORD"),
    os.environ.get("PG_HOST"),
    os.environ.get("PG_PORT")
)

samples = src.samples_loader_from_minio(
    os.environ.get('MINIO_SERVER'),
    os.environ.get('MINIO_ACCESS_KEY'),
    os.environ.get('MINIO_SECRET_KEY'),
    return_filename=True
)

#TODO: refactor code below
corpus = src.Corpus(port)
dictionary = pd.DataFrame(corpus.fetchall(),columns=["diagnostic","specialty"])
samples_df = pd.DataFrame(samples,columns=["filename","diagnostic"]).sort_values("diagnostic")
samples_specialty = samples_df.merge(dictionary,how="left")
samples_specialty_na = samples_specialty[samples_specialty.specialty.isna()]
samples_specialty_no_na = samples_specialty.dropna()
logger.info(f"in {len(samples_specialty_na)} documents the specialty was not found")
pipe = sklearn.pipeline.Pipeline([
    ("vectorizer", sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2))),
    ("classifier", sklearn.naive_bayes.ComplementNB())
])
logger.info(f"predicting not found specialties")
classifier = pipe.fit(dictionary.dropna().diagnostic,dictionary.dropna().specialty)
samples_specialty_na["specialty"] = classifier.predict(samples_specialty_na.diagnostic)
samples_specialty_predicted = pd.concat([samples_specialty_no_na,samples_specialty_na])
assert(len(samples) == len(samples_specialty_predicted))
samples_summary = pd.concat([samples_specialty_predicted.specialty.value_counts(normalize=False),samples_specialty_predicted.specialty.value_counts(normalize=True)],axis=1)
samples_summary.columns = ["n","p"]
samples_summary.index.rename("specialty",inplace=True)
assert(len(samples) == samples_summary.n.sum())
samples_specialty_predicted["corpus"] = np.where(samples_specialty_predicted.specialty.isin(src.DENTAL_SPECIALTIES),"dental","not dental")
samples_summary_corpus = pd.concat([samples_specialty_predicted.corpus.value_counts(normalize=False),samples_specialty_predicted.corpus.value_counts(normalize=True)],axis=1)
samples_summary_corpus.columns = ["n","p"]
samples_summary_corpus.index.rename("specialty",inplace=True)
assert(len(samples) == samples_summary_corpus.n.sum())
with open("specialty_mapper.json", "w", encoding="utf-8") as f:
    mapper = samples_specialty_predicted[["filename","specialty"]].set_index("filename").specialty.to_dict()
    json.dump(mapper, f, ensure_ascii=False, indent=2)
with open("summary_by_specialty.json", "w", encoding="utf-8") as f:
    summary = {
        "general": samples_summary.to_dict(orient="index"),
        "by_corpus": samples_summary_corpus.to_dict(orient="index")
    }
    json.dump(summary, f, ensure_ascii=False, indent=2)

dental_samples = [filename for filename,specialty in mapper.items() if specialty in src.DENTAL_SPECIALTIES]

d = src.Descriptor(
    samples_location="var",
    samples=[sample[1] for sample in samples]
)
d.calculate_and_write("samples_description.json")

d_d = src.Descriptor(
    samples_location="var",
    samples=[sample[1] for sample in samples if sample[0] in dental_samples]
)
d_d.calculate_and_write("samples_description_dental.json")

d_nd = src.Descriptor(
    samples_location="var",
    samples=[sample[1] for sample in samples if not sample[0] in dental_samples]
)
d_nd.calculate_and_write("samples_description_not_dental.json")