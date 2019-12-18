import pandas as pd
import os
import logging
import json

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
        
