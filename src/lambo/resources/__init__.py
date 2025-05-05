"""
The package containing resources used by LAMBO:

* ``emoji.tab`` includes a list of emoji characters recognised as special tokens. The dictionary is taken from `Morfeusz SGJP (29.05.2022) <http://morfeusz.sgjp.pl/download/>`_,
* ``pauses.txt`` lists the pause markers, also treated as special tokens,
* ``turn_regexp.txt`` lists the regular expression denoting divisions into turns,
* ``languages.txt`` is a space-spearated list of languages supported by LAMBO. Each line contains the following columns:

    * model name -- indicating the corpus a model was trained on -- usually a UD treebank,
    * language code -- according to ISO 639-1 (``?`` if no code is available)
    * language name,
    * marker of recommended model for a language -- asterisk (if recommended) or empty. USed only for languages with more than one model.
"""