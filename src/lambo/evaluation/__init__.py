"""
The module for checking the performance of a given segmenter. A segmenter can be any procedure that returns a ``Document`` based on input text. The performance is computed by comparing the returned segmentation with the gold  standard using the evaluation script from `CoNLL 2018 shared task on universal dependencies <https://universaldependencies.org/conll18/evaluation.html>`_ in ``conll18_ud_eval.py``. Note that the script is licensed through Mozilla Public License 2.0.
"""
