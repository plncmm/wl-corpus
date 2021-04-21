import numpy as np
import pandas as pd
import warnings
import itertools
import logging
import copy
import os


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
        nested_sequence = [
            item for sublist in nested_sequence for item in sublist + [['O']]]
    flat_sequence = list(itertools.chain(*nested_sequence))
    entity_set = set([e.replace("B-", "")
                      for e in flat_sequence if e.startswith("B-")])
    sequences = {e: [] for e in entity_set}
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
    overlap = range(max(entity_1[1], entity_2[1]),
                    min(entity_1[2], entity_2[2])+1)
    return overlap


def join_sentences(sentences):
    all_text = []
    for sentence in sentences:
        all_text += sentence + ['\n']
    return all_text


def get_borders(nest):
    start, end = (nest[0][1], nest[0][2])
    for e in nest:
        if e[1] < start:
            start = e[1]
        if e[2] > end:
            end = e[2]
    return start, end


def get_inconsistencies(nest, nests):
    start, end = get_borders(nest)
    inconsistencies = []
    for n in nests:
        current_start, current_end = get_borders(n)
        if (current_start > start) & (current_end > end):
            break
        elif ((current_start >= start) & (current_start <= end)) | ((current_end >= start) & (current_end <= end)):
            inconsistencies.append(n)
    return flatten(inconsistencies)


def nest_entities(entities):
    nests = []
    last = 0
    for e in entities:
        start, end = (e[1], e[2])
        if (start > last) | (last == 0):
            nest = [e]
            for e2 in entities:
                current_start, current_end = (e2[1], e2[2])
                if ((current_start >= start) & (current_start <= end)) | ((current_end >= start) & (current_end <= end)):
                    if not e2 in nest:
                        nest.append(e2)
                    if current_start < start:
                        start = current_start
                    if current_end > end:
                        end = current_end
                    if current_end > last:
                        last = current_end
            nests.append(nest)
    return nests


def check_wrong_label_right_span(entity, inconsistencies):
    result = []
    name, start, end = entity
    for i in inconsistencies:
        current_name, current_start, current_end = i
        if (name != current_name) & ((start, end) == (current_start, current_end)):
            result.append(i)
    return result


def check_wrong_label_overlapping_span(entity, inconsistencies):
    result = []
    name, start, end = entity
    for i in inconsistencies:
        current_name, current_start, current_end = i
        if (len(set(range(start, end+1)).intersection(range(current_start, current_end+1))) > 0) & (name != current_name) & ((start, end) != (current_start, current_end)):
            result.append(i)
    return result


def check_right_label_overlapping_span(entity, inconsistencies):
    result = []
    name, start, end = entity
    for i in inconsistencies:
        current_name, current_start, current_end = i
        if (len(set(range(start, end+1)).intersection(range(current_start, current_end+1))) > 0) & (name == current_name) & ((start, end) != (current_start, current_end)):
            result.append(i)
    return result


def check_false(entity, inconsistencies):
    result = []
    if not entity in inconsistencies:
        result = [entity]
    return result


def flatten(t): return [item for sublist in t for item in sublist]


def ner_report(y_true, y_pred):
    true_entities = get_entities_from_nested_sequence(y_true)
    pred_entities = get_entities_from_nested_sequence(y_pred)
    true_entities = sorted(true_entities, key=lambda tup: tup[1])
    pred_entities = sorted(pred_entities, key=lambda tup: tup[1])

    true_nested_entities = nest_entities(true_entities)
    pred_nested_entities = nest_entities(pred_entities)

    correct = []
    right_label_overlapping_span = []
    wrong_label_overlapping_span = []
    wrong_label_right_span = []
    false_positive = []
    false_negative = []

    for t in true_nested_entities:
        inconsistencies = get_inconsistencies(t, pred_nested_entities)
        logging.debug(f"true nest: {t}, inconsistencies: {inconsistencies}")
        t_ = t[:]
        for t2 in t_:
            if t2 in inconsistencies:
                logging.debug(f"{t2} correct")
                correct.append(t2)
                t.pop(t.index(t2))
                inconsistencies.pop(inconsistencies.index(t2))
        t_ = t[:]
        for t2 in t_:
            current_right_label_overlapping_span = check_right_label_overlapping_span(
                t2, inconsistencies)
            if current_right_label_overlapping_span:
                logging.debug(f"{t2} {current_right_label_overlapping_span} right_label_overlapping_span")
                for a in current_right_label_overlapping_span:
                    right_label_overlapping_span.append((t2, a))
                t.pop(t.index(t2))
                for crlos in current_right_label_overlapping_span:
                    inconsistencies.pop(inconsistencies.index(crlos))
        t_ = t[:]
        for t2 in t_:
            current_wrong_label_overlapping_span = check_wrong_label_overlapping_span(
                t2, inconsistencies)
            if current_wrong_label_overlapping_span:
                logging.debug(f"{t2} {current_wrong_label_overlapping_span} wrong_label_overlapping_span")
                for a in current_wrong_label_overlapping_span:
                    wrong_label_overlapping_span.append((t2, a))
                t.pop(t.index(t2))
                for cwlos in current_wrong_label_overlapping_span:
                    inconsistencies.pop(inconsistencies.index(cwlos))
        for t2 in t_:
            current_wrong_label_right_span = check_wrong_label_right_span(
                t2, inconsistencies)
            if current_wrong_label_right_span:
                logging.debug(f"{t2} {current_wrong_label_right_span} wrong_label_right_span")
                for a in current_wrong_label_right_span:
                    wrong_label_right_span.append((t2, a))
                t.pop(t.index(t2))
                for cwlrs in current_wrong_label_right_span:
                    inconsistencies.pop(inconsistencies.index(cwlrs))
        t_ = t[:]
        for t2 in t_:
            current_false_negative = check_false(
                t2, inconsistencies)
            if current_false_negative:
                logging.debug(f"{t2} false_negative")
                for a in current_false_negative:
                    false_negative.append(t2)
                t.pop(t.index(t2))
    rest = set(flatten(pred_nested_entities)) - set(correct) - set(false_negative) - set(flatten(wrong_label_right_span)
                                                                                         ) - set(flatten(wrong_label_overlapping_span)) - set(flatten(right_label_overlapping_span))
    for r in list(rest):
        logging.debug(f"{r} false_positive")
        false_positive.append(r)
    report = {
        "n_exact_matches": len(correct),
        "n_false_positives": len(false_positive),
        "n_false_negatives": len(false_negative),
        "n_wrong_label_right_span": len(wrong_label_right_span),
        "n_wrong_label_overlapping_span": len(wrong_label_overlapping_span),
        "n_right_label_overlapping_span": len(right_label_overlapping_span),
        "correct": (correct),
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "wrong_label_right_span": wrong_label_right_span,
        "wrong_label_overlapping_span": wrong_label_overlapping_span,
        "right_label_overlapping_span": right_label_overlapping_span,
    }
    return report

def read_separate_predictions(predictions_path):
    annotations = {}
    for entity in os.listdir(predictions_path):
        entity_path = predictions_path / entity
        entity_name = entity_path.stem
        annotations[entity_name] = []
        with open(entity_path, "r", encoding="utf-8") as f:
            sentence = []
            true = []
            predicted = []
            for line in f:
                line = line.rstrip()
                if line != "":
                    token, true_entity, predicted_entity = line.split(" ")
                    sentence.append(token)
                    true.append([true_entity])
                    predicted.append([predicted_entity])
                else:
                    annotations[entity_name].append({
                        "sentence":sentence,
                        "true":true,
                        "predicted":predicted
                    })
                    sentence = []
                    true = []
                    predicted = []

    annotations_consolidated = []
    for i,(_,val) in enumerate(annotations.items()):
        if i == 0:
            annotations_consolidated = copy.deepcopy(val)
        else:
            for j,s in enumerate(val):
                for k in range(len(s["sentence"])):
                    current_true_annotation = annotations_consolidated[j]["true"][k]
                    new_true_annotation = s["true"][k]
                    if new_true_annotation[0] == "O":
                        pass
                    elif "O" in current_true_annotation:
                        current_true_annotation.extend(new_true_annotation)
                        current_true_annotation.pop(current_true_annotation.index("O"))
                    else:
                        current_true_annotation.extend(new_true_annotation)

                    current_predicted_annotation = annotations_consolidated[j]["predicted"][k]
                    new_predicted_annotation = s["predicted"][k]
                    if new_predicted_annotation[0] == "O":
                        pass
                    elif "O" in current_predicted_annotation:
                        current_predicted_annotation.extend(new_predicted_annotation)
                        current_predicted_annotation.pop(current_predicted_annotation.index("O"))
                    else:
                        current_predicted_annotation.extend(new_predicted_annotation)
    y_trues = [annotations_consolidated[i]["true"] for i in range(len(annotations_consolidated))]
    y_predicteds = [annotations_consolidated[i]["predicted"] for i in range(len(annotations_consolidated))]
    sentences = [annotations_consolidated[i]["sentence"] for i in range(len(annotations_consolidated))]
    return y_trues, y_predicteds, sentences