"""Find sentence level accuracies."""
from argparse import ArgumentParser
import numpy as np


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def create_sentence_level_tags(conll_lines):
    """Create sentence level tags from conll formatted text."""
    all_sentence_tags = list()
    sentence_tags = list()
    for line in conll_lines:
        line = line.strip()
        if line:
            sentence_tags.append(line)
        else:
            all_sentence_tags.append(sentence_tags)
            sentence_tags = list()
    else:
        if sentence_tags:
            all_sentence_tags.append(sentence_tags)
            sentence_tags = list()
    return all_sentence_tags


def find_accuracies(gold, predicted):
    """Find 2 types of accuracies: strict accuracis between gold and predicted, 1 if complete match else 0, relaxed denoted percentage match."""
    strict, relaxed = list(), list()
    for gold_item, predicted_item in zip(gold, predicted):
        assert len(gold_item) == len(predicted_item)
        if gold_item == predicted_item:
            strict.append(1.)
            relaxed.append(1.)
        else:
            strict.append(0)
            relaxed.append(np.mean(np.array(gold_item) == np.array(predicted_item)))
    return strict, relaxed


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser(description="Find strict and relaxed sentence level accuracies.")
    parser.add_argument('--gold', dest='g', help="Enter the gold labels file")
    parser.add_argument('--pred', dest='p', help="Enter the predicted labels file")
    args = parser.parse_args()
    gold_lines = read_lines_from_file(args.g)
    pred_lines = read_lines_from_file(args.p)
    gold_sentence_tags = create_sentence_level_tags(gold_lines)
    pred_sentence_tags = create_sentence_level_tags(pred_lines)
    strict, relaxed = find_accuracies(gold_sentence_tags, pred_sentence_tags)
    mean_strict = np.mean(strict)
    mean_relaxed = np.mean(relaxed)
    print('Sentence Accuracies strict={} and relaxed={}'.format(mean_strict, mean_relaxed))


if __name__ == '__main__':
    main()
