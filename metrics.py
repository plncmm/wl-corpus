import numpy as np
import pandas as pd 
# from seqeval.metrics.sequence_labeling import get_entities
# from entity_classifier_model import EntityIdentifier
import warnings
import itertools

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end

def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def get_entities_from_nested_sequence(nested_sequence):
    if any(isinstance(s, list) for s in nested_sequence):
        nested_sequence = [item for sublist in nested_sequence for item in sublist + [['O']]]
    flat_sequence = list(itertools.chain(*nested_sequence))
    entity_set = set([e.replace("B-","") for e in flat_sequence if e.startswith("B-")])
    sequences = {e:[] for e in entity_set}
    for entity in entity_set:
        for element in nested_sequence:
            appendable_element = "O"
            for subelement in element:
                if entity in subelement:
                    appendable_element = subelement
                    break
            sequences[entity].append(appendable_element)
    entities = []
    for s in sequences.values():
        entities.append(get_entities(s))
    return list(itertools.chain(*entities))

def get_overlap(entity_1, entity_2):
  overlap = range(max(entity_1[1], entity_2[1]), min(entity_1[2], entity_2[2])+1)
  return  overlap

def join_sentences(sentences):
  all_text = []
  for sentence in sentences:
    all_text += sentence +['\n']
  return all_text

def get_error_types(y_true,y_pred):
  if isinstance(y_true[0],list):
    true_entities = get_entities_from_nested_sequence(y_true)
    pred_entities = get_entities_from_nested_sequence(y_pred)
  else:
    true_entities = get_entities(y_true)
    pred_entities = get_entities(y_pred)
  correct = set(true_entities)&set(pred_entities)
  true_entities_rest = set(true_entities)-correct
  pred_entities_rest = set(pred_entities)-correct

  right_label_overlapping_span = []
  wrong_label_overlapping_span = []
  wrong_label_right_span= []
  complete_false_positive = []
  complete_false_negative = []
  for true_entity in list(true_entities_rest):
    for pred_entity in list(pred_entities_rest):
      overlap = get_overlap(true_entity, pred_entity)
      if len(overlap)>0:
        if true_entity[0]==pred_entity[0]:
          right_label_overlapping_span.append((true_entity, pred_entity))
        elif (true_entity[1]==pred_entity[1]) & (true_entity[2]==pred_entity[2]):
          wrong_label_right_span.append((true_entity, pred_entity))
        else:
          wrong_label_overlapping_span.append((true_entity, pred_entity))

  complete_false_positive = pred_entities_rest - set([item[1] for item in right_label_overlapping_span])-\
                                                set([item[1] for item in wrong_label_overlapping_span])-set([item[1] for item in wrong_label_right_span]) 

  complete_false_negative = true_entities_rest - set([item[0] for item in right_label_overlapping_span])-\
                                                set([item[0] for item in wrong_label_overlapping_span])-set([item[0] for item in wrong_label_right_span])
  return  correct, right_label_overlapping_span, wrong_label_overlapping_span, wrong_label_right_span, complete_false_positive, complete_false_negative

class results_analyser:
  def __init__(self, y_true, y_pred, sentences, 
  # entity_classifier_model, precisions
  ):
    self.y_true = y_true
    self.y_pred = y_pred
    self.sentences = sentences
    # self.model = entity_classifier_model
    correct,right_label_over_span, wrong_label_over_span, wrong_label_right_span, false_positive, false_negative = get_error_types(y_true, y_pred)
    self.correct = correct
    self.right_label_over_span = right_label_over_span
    self.wrong_label_over_span = wrong_label_over_span
    self.wrong_label_right_span = wrong_label_right_span
   
    self.n_correct = len(correct)
    self.n_right_label_over_span = len(right_label_over_span)
    self.n_wrong_label_over_span = len(wrong_label_over_span)
    self.n_wrong_label_right_span = len(wrong_label_right_span)
    self.n_false_positive = len(false_positive)
    self.n_false_negative = len(false_negative) 
    # self.precisions = precisions
    self.joined_sentences = join_sentences(self.sentences)
    self.sentence_boundaries = [i for i, x in enumerate(self.joined_sentences) if x == "\n"]
  
  def get_sentence_index(self,entity):
    sentence_index = [i+1 for i in range(len(self.sentence_boundaries)) if (entity[1]>self.sentence_boundaries[i]) and (entity[2]<self.sentence_boundaries[i+1])]
    return sentence_index[0]

  def get_phrase(self,entity):
     phrase = ' '.join(self.joined_sentences[entity[1]:entity[2]+1])
     return phrase

  def get_correct_phrases(self):
    
    correct_data = list()
    for true_entity in self.correct:
      correct_data.append((true_entity[0], self.get_phrase(true_entity)))
    return correct_data

  def get_raw_type_5_data(self):
    
    Type_5_phrase = list()
    for true_entity, pred_entity in self.right_label_over_span:
      Type_5_phrase.append((pred_entity[0], self.get_phrase(true_entity),self.get_phrase(pred_entity)))
    return Type_5_phrase
    
  # def get_processed_type_5_data(self):
  #   Type_5_phrase = self.get_raw_type_5_data()
  #   probs, id_2_tag = self.model.predict([item[2] for item in Type_5_phrase], predict_batch_size = 128)
  #   processed_Type_5_phrase = list()
  #   for item,prob in zip(Type_5_phrase,probs):
  #     prediction_id = int(np.array(prob.argmax()))
  #     prediction_probability = float("{0:.3f}".format(np.array(prob.max())))
  #     processed_Type_5_phrase.append(item +(prediction_id,prediction_probability))  
  #   return processed_Type_5_phrase, id_2_tag 


  def get_to_annotate(self):
    to_annotate = pd.DataFrame(columns = ['sentence','extracted entity', 'tag','predicted entity'])
    sentences = list()
    extracted_entities = list()
    tags = list()
    predicted_entities = list()
    for true_entity, pred_entity in self.right_label_over_span:
      sentences.append(self.sentences[self.get_sentence_index(true_entity)])
      extracted_entities.append(self.get_phrase(true_entity))
      tags.append(true_entity[0])
      predicted_entities.append(self.get_phrase(pred_entity))

    to_annotate['sentence'] = sentences
    to_annotate['extracted entity'] = extracted_entities
    to_annotate['tag'] = tags
    to_annotate['predicted entity'] = predicted_entities

    return to_annotate


  # def get_overlap_score(self, prob = False):
  #   processed_Type_5_phrase , id_2_tag = self.get_processed_type_5_data()
  #   if prob:
  #     overlap_pred_score= np.sum([item[4] for item in processed_Type_5_phrase if item[0]==id_2_tag[int(item[3])]])  #*self.precisions[int(item[3])]
  #   else:
  #     overlap_pred_score= np.sum([1 for item in processed_Type_5_phrase if item[0]==id_2_tag[int(item[3])]])
  
  #   return overlap_pred_score

  def get_all_fscores(self):
    n_correct = len(self.correct)
    if isinstance(self.y_true[0],list):
      true_entities = get_entities_from_nested_sequence(self.y_true)
      pred_entities = get_entities_from_nested_sequence(self.y_pred)
    else:
      true_entities = get_entities(self.y_true)
      pred_entities = get_entities(self.y_pred)
    n_true = len(true_entities)
    n_pred = len(pred_entities)

    
    p = (n_correct ) / n_pred if n_pred > 0 else 0
    r = n_correct / n_true if n_true > 0 else 0
    exact_f_score = 2 * p * r / (p + r) if p + r > 0 else 0



    p = (n_correct + len((self.right_label_over_span)) ) / n_pred if n_pred > 0 else 0
    r = (n_correct + len((self.right_label_over_span)) )/ n_true if n_true > 0 else 0
    relaxed_f_score = 2 * p * r / (p + r) if p + r > 0 else 0

    # overlap_pred_score = self.get_overlap_score()
    # p = (n_correct + overlap_pred_score) / n_pred if n_pred > 0 else 0
    # r = (n_correct + overlap_pred_score) / n_true if n_true > 0 else 0
    # user_exp_f_score = 2 * p * r / (p + r) if p + r > 0 else 0
    exact_f_score, relaxed_f_score, #user_exp_f_score

    return exact_f_score, relaxed_f_score, #user_exp_f_score