from lambo.evaluation.conll18_ud_eval import load_conllu, evaluate, UDError
from lambo.utils.printer import print_document_to_conll


def evaluate_segmenter(segmenter, test_text, gold_path, tmp_path):
    """
    Check the performance of a given segmenter by comparing its output to gold standard on a given text.
    
    :param segmenter: segmenter to be checked
    :param test_text: test data string
    :param gold_path: path to the gold standard in .conllu format
    :param tmp_path: temporary path to write the segmenter output in .conllu format
    :return: a dictionary with results, containing entries for ``Tokens``, ``Words`` and ``Sentences``, the value of each being a dictionary with ``F1``, ``precision`` and ``recall``.
    """
    result = {}
    document = segmenter.segment(test_text)
    print_document_to_conll(document, tmp_path)
    with open(tmp_path) as fPred:
        with open(gold_path) as fGold:
            pred = load_conllu(fPred)
            gold = load_conllu(fGold)
            try:
                conll_result = evaluate(gold, pred)
                for category in ['Tokens', 'Words', 'Sentences']:
                    result[category] = {'F1': conll_result[category].f1, 'precision': conll_result[category].precision,
                                        'recall': conll_result[category].recall}
            except UDError as e:
                for category in ['Tokens', 'Words', 'Sentences']:
                    result[category] = {'F1': 0.0, 'precision': 0.0,
                                        'recall': 0.0}
    return result
