import pandas as pd
import os
import logging
import json
import random
import time
import hashlib
import sshtunnel
import statistics
import scipy.stats
import dotenv
from spacy.lang.es import Spanish
nlp = Spanish()
spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)
dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DENTAL_SPECIALTIES = [
    "ENDODONCIA",
    "OPERATORIA",
    "ENDODONCIA",
    "REHABILITACION: PROTESIS FIJA",
    "CIRUGIA MAXILO FACIAL",
    "ODONTOLOGIA INDIFERENCIADO",
    "REHABILITACION: PROTESIS REMOVIBLE",
    "TRASTORNOS TEMPOROMANDIBULARES Y DOLOR OROFACIAL",
    "CIRUGIA BUCAL",
    "PERIODONCIA"
]

def mean_confidence_interval(data, confidence=0.95):
    """
    Given a list of numbers, computes the confidence interval of the sample.

    Args:
    data: List of numerix data
    confidence: float, confidence level

    Returns:
    tuple of the mean of the data, lower limit and upper limit.
    """
    n = len(data)
    m, se = statistics.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def sample_filenames_from_dir(directory):
    """
    Given a file path, returns a list of .txt filenames in that directory.
    """
    samples_filenames = [directory + filename for filename in os.listdir(directory) if filename.endswith(".txt")]
    return samples_filenames

def samples_loader(filenames):
    """
    From a list of file paths, returns a list of the text content of each file.
    """
    current_samples = []
    for sample in filenames:
        try:
            with open(sample, "r", encoding="utf-8") as samplefile:
                current_samples.append(samplefile.read())
        except UnicodeDecodeError:
            with open(sample, "r", encoding="latin-1") as samplefile:
                current_samples.append(samplefile.read())
        
    return current_samples

def tokenizer(document):
    """
    From a string, returns a list of tokens.
    """
    result = list(spacy_tokenizer(document))
    result = [str(token) for token in result]
    return result

def load_corpus_from_dw(specialties = []):
    with sshtunnel.open_tunnel((os.environ.get("TUNNEL_HOST"), int(os.environ.get("TUNNEL_PORT"))),
         ssh_username=os.environ.get("TUNNEL_USER"),
         ssh_password=os.environ.get("TUNNEL_PASSWORD"),
         remote_bind_address=(os.environ.get("PG_HOST"), int(os.environ.get("PG_PORT")))) as server:
        if specialties:
            specialties_condition = ",".join(f"'{specialty}'" for specialty in specialties)
            specialties_condition = f"AND especialidad in ({specialties_condition})"
        else:
            specialties_condition = ""
        query = f"""
        
    SELECT data."Sospecha diagnóstica" AS document
    FROM 
        data
    WHERE
        length(data."Sospecha diagnóstica") > 100
        {specialties_condition}
    GROUP BY
        data."Sospecha diagnóstica"
    ORDER BY
        random()
        
        """
        logger.info("fetching data from data warehouse")
        corpus = pd.read_sql_query(query,f'postgresql://{os.environ.get("PG_USER")}:{os.environ.get("PG_PASSWORD")}@127.0.0.1:{server.local_bind_port}/wl').document.tolist()
        return corpus

class WlTextRawLoader:
    """
    Class to construct a text corpus from a csv.
    """
    def __init__(self, raw_data_directory):
        """
        Receives a directory path containing a bunch of csv.
        """
        self.filenames = [raw_data_directory + filename for filename in os.listdir(raw_data_directory)]
    def load_files(self,column_name="SOSPECHA_DIAG",min_document_length=100):
        """
        Load the csv files into a consolidated list of text data, specifying the column that contains the text data and the minimum document length to accept.
        """
        self.corpus = []
        for filename in self.filenames:
            logger.info(filename)
            current = pd.read_csv(filename, low_memory=False)
            current_documents = set(current[column_name].map(str).tolist())
            for document in current_documents:
                if len(document) > min_document_length:
                    self.corpus.append(document)
            self.corpus = list(set(self.corpus))
    def export(self, destination_file):
        """
        Saves the current list of documents as a JSON file.
        """
        with open(destination_file, "w", encoding="utf-8") as jsonfile:
            json.dump(self.corpus, jsonfile, ensure_ascii=False, indent=4)
        
class SamplePicker:
    """
    Class to create a sample picker object to pick random documents from a corpus.
    """
    def __init__(self,samples_location,samples_rejected_location,corpus_location,corpus='both'):
        """
        Constructs a sample picker.

        Args:
        corpus_location: Location of the corpus as a json file.
        samples_location: Directory path for the directory where the past picked samples are stored.
        samples_rejected_location: Directory path for the directory where rejected samples are stored.
        """
        if corpus_location != "dw":
            if corpus == 'both':
                with open(corpus_location, encoding="utf-8") as json_file:
                    self.corpus = json.load(json_file)
            else:
                raise NotImplementedError
        else:
            if corpus == "dental":
                self.corpus = load_corpus_from_dw(DENTAL_SPECIALTIES)
            elif corpus == "both":
                self.corpus = load_corpus_from_dw()
            else:
                raise NotImplementedError
        logger.info("corpus size: {} documents".format(len(self.corpus)))
        self.samples_location = samples_location
        self.samples_filenames = sample_filenames_from_dir(samples_location)
        self.samples_rejected_filenames = sample_filenames_from_dir(samples_rejected_location)
        self.samples_filenames = self.samples_filenames + self.samples_rejected_filenames
        self.current_samples = samples_loader(self.samples_filenames)
    def pick(self,n):
        """
        Pick n samples from the corpus.
        """
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

class Descriptor:
    """
    Class to create a text corpus descriptor.
    """
    def __init__(self,samples_location):
        """
        Construct the descriptor given a directory path to the directory which contains the text samples as txt.
        """
        self.samples_filenames = sample_filenames_from_dir(samples_location)
        self.samples = samples_loader(self.samples_filenames)
    def calculate_and_write(self,report_location):
        """
        Calculate the description metrics and writes a report as a json file.
        """
        self.samples_tokenized = [tokenizer(sample) for sample in self.samples]
        self.vocab = list(set([word for document in self.samples_tokenized for word in document]))
        self.tokens_n = [len(document) for document in self.samples_tokenized]
        self.normal_test = scipy.stats.shapiro(self.tokens_n)
        self.report = {
            "documents_n":len(self.samples),
            "tokens_n_sum": sum(self.tokens_n),
            "tokens_n_mean_ci": mean_confidence_interval(self.tokens_n),
            "tokens_normal_dis": True if self.normal_test[1] < 0.05 else False,
            "tokens_shapiro_p": self.normal_test[1],
            "tokens_n_sd": statistics.stdev(self.tokens_n),
            "tokens_n_var": statistics.variance(self.tokens_n),
            "tokens_n_median": statistics.median(self.tokens_n),
            "vocab_n": len(self.vocab),
            "vocab": self.vocab,
            "tokens_n": self.tokens_n,
            "samples_tokenized": self.samples_tokenized,
            "samples": self.samples
        }
        with open(report_location, "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)

def move_file(from_folder,to_folder,filename):
    """
    Moves a file from a folder to another folder.
    """
    try:
        os.replace(from_folder+filename,to_folder+filename)
        logger.info(from_folder+filename + " moved")
    except FileNotFoundError:
        logger.error(from_folder+filename+" not found")
class Discarder:
    """
    Class to create a file discarder.
    """
    def __init__(self,from_folder="samples/",to_folder="samples_rejected/"):
        """
        Constructor which receives a from folder and a to folder.
        """
        self.file_list = []
        self.from_folder = from_folder
        self.to_folder = to_folder
    def from_txt(self, location):
        """
        Receives a filepath to a txt which contains a list of files to discard separated by line break.
        """
        with open(location, encoding="utf-8", mode="r") as f:
            for line in f:
                line = line.rstrip()
                if line.endswith(".txt"):
                    self.file_list.append(line)
                else:
                    self.file_list.append(line+".txt")
    def discard(self):
        """
        Executes a discarding process.
        """
        for filename in self.file_list:
            move_file(self.from_folder,self.to_folder,filename)