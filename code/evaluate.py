""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    preds = []
    for question in dataset:
        total += 1
        if question['id'] not in predictions:
            message = 'Unanswered question ' + question['id'] + \
                        ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = [question['answer']['text'][0]]
        prediction = predictions[question['id']]['prediction']
        exact_match = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        # f1 += metric_max_over_ground_truths(
        #     f1_score, prediction, ground_truths)

        preds.append({
            'expected_prediction': ground_truths[0],
            'prediction': prediction,
            'correct': exact_match,
            'maxProb': predictions[question['id']]['maxProb']
        })

    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total

    # return {'exact_match': exact_match, 'f1': f1}

    return preds

def create_evaluation_files(dataset_file,prediction_file,out):
    dataset = []
    with open(dataset_file) as f:
        
        for line in f:
            dataset.append(json.loads(line))
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    with open(out, 'w') as outfile:
        json.dump(evaluate(dataset, predictions), outfile, indent=4)


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    # parser.add_argument('dataset_file', help='Dataset file')
    # parser.add_argument('prediction_file', help='Prediction File')
    dataset_files = ["dataset/snli_squad_eval.json","dataset/swag_squad_eval.json", "dataset/csqa_squad_eval.json", "dataset/anli_squad_eval.json", "dataset/siqa_squad_eval.json"]
    hetero_prediction_files = ["output-hetero/eval_snli_squad_eval_predictions.json", "output-hetero/eval_swag_squad_eval_predictions.json", "output-hetero/eval_csqa_squad_eval_predictions.json", "output-hetero/eval_anli_squad_eval_predictions.json", "output-hetero/eval_siqa_squad_eval_predictions.json"]
    homo_prediction_files = ["output-homo/eval_snli_squad_eval_predictions.json", "output-homo/eval_swag_squad_eval_predictions.json", "output-homo/eval_csqa_squad_eval_predictions.json", "output-homo/eval_anli_squad_eval_predictions.json", "output-homo/eval_siqa_squad_eval_predictions.json"]
    args = parser.parse_args()
    for i,dataset in enumerate(dataset_files):
        print(dataset)
        create_evaluation_files(dataset,hetero_prediction_files[i],dataset.split('/')[1].split('_')[0] + "_eval_prob_n_preds_hetero.json")
        create_evaluation_files(dataset,homo_prediction_files[i],dataset.split('/')[1].split('_')[0] + "_eval_prob_n_preds_homo.json")




    