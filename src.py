import pandas as pd
import os
import logging
import json
import random
import time
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WlTextRawLoader:
    def __init__(self, raw_data_directory):
        self.filenames = [raw_data_directory + filename for filename in os.listdir(raw_data_directory)]
    def load_files(self):
        self.corpus = []
        for filename in self.filenames:
            logger.info(filename)
            current = pd.read_csv(filename, low_memory=False)
            current_documents = set(current['SOSPECHA_DIAG'].map(str).tolist())
            for document in current_documents:
                if len(document) > 100:
                    self.corpus.append(document)
            self.corpus = list(set(self.corpus))
    def export(self, destination_file):
        with open(destination_file, "w", encoding="utf-8") as jsonfile:
            json.dump(self.corpus, jsonfile, ensure_ascii=False, indent=4)
        
class SamplePicker:
    def __init__(self,corpus_location,samples_location,samples_rejected_location):
        self.samples_location = samples_location
        with open(corpus_location) as json_file:
            self.corpus = json.load(json_file)
        self.samples_filenames = [samples_location + filename for filename in os.listdir(samples_location)]
        self.samples_rejected_filenames = [samples_rejected_location + filename for filename in os.listdir(samples_rejected_location)]
        self.samples_filenames = self.samples_filenames + self.samples_rejected_filenames
        self.current_samples = []
        for sample in self.samples_filenames:
            with open(sample, "r", encoding="utf-8") as samplefile:
                self.current_samples.append(samplefile.read())
    def pick(self,n):
        self.picked_samples = []
        while len(self.picked_samples) < n:
            remaining_samples_n = n - len(self.picked_samples)
            self.picked_samples.extend(random.choices(self.corpus,k=remaining_samples_n))
            self.picked_samples = [sample for sample in self.picked_samples if sample not in self.current_samples]
            logger.info("picked {} samples".format(len(self.picked_samples)))
        for sample in self.picked_samples:
            filename = hashlib.md5(sample.encode("utf-8")).hexdigest()
            with open(self.samples_location + filename + ".txt", "w", encoding="utf-8") as textfile:
                textfile.write(sample)