**UPDATE 15.03.2024**: *The 2.3 update includes preprocessing improved to deal with unexpected characters (e.g. in
foreign names) and all models re-trained using [UD 2.13](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5287).
LAMBO now supports 67 languages.*

# LAMBO segmenter

LAMBO (Layered Approach to Multi-level BOundary identification) is a segmentation tool that is able to divide text on several levels:
1. Dividing the original text into *turns* according to the provided list of separators. Turns can correspond to seperate utterences in a dialogue, paragraphs in a continuous text, etc.
2. Splitting each turn into *sentences*.
3. Finding *tokens* in sentences. Most tokens correspond to words. LAMBO also supports special tokens that should be kept separate regardless of context, such as emojis and pause markers.
4. Splitting tokens that are detected to be *multi-word* into *sub-words* (for selected languages).

LAMBO is a machine learning model, which means it was trained to recognise boundaries of tokens and sentences from real-world text. It is implemented as a [PyTorch](https://pytorch.org/) deep neural network, including embeddings and recurrent layers operating at the character level. At the same time, LAMBO contains rule-based elements to allow a user to easily adjust it to one's needs, e.g. by adding custom special tokens or turn division markers.

LAMBO was developed in context of dependency parsing. Thus, it includes models trained on [Universal Dependencies treebanks](https://universaldependencies.org/#language-), uses `.conllu` as the training [data format](https://universaldependencies.org/conll18/evaluation.html) and supports integration with [COMBO](https://gitlab.clarin-pl.eu/syntactic-tools/combo), a state-of-the-art system for dependency parsing and more. However, you can use LAMBO as the first stage of any NLP process.

LAMBO currently includes models trained on 130 corpora in 67 languages. The full list is available in [`languages.txt`](src/lambo/resources/languages.txt). Most of these are pretrained on unsupervised masked character prediction using multilingual corpora from [OSCAR](https://oscar-corpus.com/) and fine-tuned on UD 2.13 corpus.

For 54 of the corpora, a subword splitting model is available. Note that different types of multi-word tokens exist in different languages:
- those that are a concatenation of their subwords, as in English: *don't* = *do* + *n't*
- those that differ from their subwords, as in Spanish: *al* = *a* + *el*

The availability and type of subword splitting model depends on the training data (i.e., UD treebank).

## Installation

Installation of LAMBO is easy.

1. First, you need to prepare an environment with Python, at least 3.10,

2. Then, install LAMBO as follows:
```
pip install --index-url https://pypi.clarin-pl.eu/ lambo
```

You now have LAMBO installed in your environment.

## Using LAMBO

To use LAMBO, you first need to import it:
```
from lambo.segmenter.lambo import Lambo
```

Now you need to create a segmenter by providing the language your text is in, e.g. `English`:
```
lambo = Lambo.get('English')
```
This will (if necessary) download the appropriate model from the online repository and load it. Note that you can use any language name (e.g. `Ancient_Greek`) or ISO 639-1 code (e.g. `fi`) from [`languages.txt`](src/lambo/resources/languages.txt).

Alternatively, you can select a specific model by defining LAMBO variant (`LAMBO_213` is the newest one) and training dataset from [`languages.txt`](src/lambo/resources/languages.txt):
```
lambo = Lambo.get('LAMBO_213-UD_Polish-PDB')
```

There are two optional arguments to the `get()` function:
- You can opt out of using subword splitter by providing `with_splitter=False`.
- You can point to a specific pyTorch device by providing `device` parameter, for example `device=torch.device('cuda')` to enable GPU acceleration.

Once the model is ready, you can perform segmentation of a given text:
```
text = "Simple sentences can't be enough... Some of us just ❤️ emojis. They should be tokens even when (yy) containing many characters, such as 👍🏿."
document = lambo.segment(text)
```

The `document` object contains a list of `turns`, each composed of `sentences`, which in turn include `tokens`. The structured could be explored in the following way:
```
for turn in document.turns:
    print('======= TURN =======')
    print('TEXT: ' + turn.text[:100] + '...')
    for sentence in turn.sentences:
        print('======= SENTENCE =======')
        print('TEXT: "' + sentence.text + '"')
        formatted = ''
        for token in sentence.tokens:
            if token.is_multi_word:
               formatted += '(' + token.text+ '=' + '-'.join(token.subwords) + ')'
            else:
               formatted += '(' + token.text + ')'
        print('TOKENS: ' + formatted)
```
This should produce the following output:
```
======= TURN =======
TEXT: Simple sentences can't be enough... Some of us just ❤️ emojis. They should be tokens even when (yy) ...
======= SENTENCE =======
TEXT: "Simple sentences can't be enough... "
TOKENS: (Simple)(sentences)(can't=ca-n't)(be)(enough)(...)
======= SENTENCE =======
TEXT: "Some of us just ❤️ emojis. "
TOKENS: (Some)(of)(us)(just)(❤️)(emojis)(.)
======= SENTENCE =======
TEXT: "They should be tokens even when (yy) containing many characters, such as 👍🏿."
TOKENS: (They)(should)(be)(tokens)(even)(when)((yy))(containing)(many)(characters)(,)(such)(as)(👍🏿)(.)
```
Note how *can't* was split and the special tokens, i.e. emojis and pause (`(yy)`) were properly recognised.

## Using LAMBO with COMBO

You can use LAMBO to segment text that is going to be processed by COMBO. To do that, first you will need to install both COMBO and LAMBO in the same environment. Make sure you have the [COMBO](https://gitlab.clarin-pl.eu/syntactic-tools/combo) version supporting LAMBO.

Once both tools are ready, you need to import COMBO with LAMBO tokeniser:
```
from combo.predict import COMBO
from combo.utils import lambo
```

You can now create COMBO instance using LAMBO for segmentation. Make sure both use models appropriate for the language of text:
```
nlp_new = COMBO.from_pretrained("polish-herbert-base-ud29",tokenizer=lambo.LamboTokenizer("pl"))
```

Now, there are two ways for interacting with COMBO-LAMBO duet. You can provide a list of sentences as input and LAMBO will tokenise each of them and pass to COMBO for further processing:
```
text = ["To zdanie jest OK 👍🏿.", "To jest drugie zdanie."]
sentences = nlp_new(text)
```

Alternatively, you can provide a single string, using the fact that LAMBO is more than a tokeniser and can split sentences on its own:
```
text="To zdanie jest OK 👍🏿. To jest drugie zdanie."
sentences = nlp_new(text)
```

In any case, you should get a full dependency parsing output, which you can print via:
```
print("{:5} {:15} {:15} {:10} {:10} {:10}".format('ID', 'TOKEN', 'LEMMA', 'UPOS', 'HEAD', 'DEPREL'))
	for sentence in sentences:
		for token in sentence.tokens:
			print("{:5} {:15} {:15} {:10} {:10} {:10}".format(str(token.id), token.token, token.lemma, token.upostag, str(token.head), token.deprel))
		print("\n")
```


## Extending LAMBO

You don't have to rely on the models trained so far in COMBO. You can use the included code to train on new corpora and languages, tune to specific usecases or simply retrain larger models with more resources. The scripts in [`examples`](src/lambo/examples) include examples on how to do that:
- `run_training.py` -- train simple LAMBO models. This script was used with [UD treebanks](https://universaldependencies.org/#language-) to generate `LAMBO_no_pretraining` models.
- `run_pretraining.py` -- pretrain unsupervised LAMBO models. This script was used with [OSCAR](https://oscar-corpus.com/).
- `run_training_pretrained.py` -- train LAMBO models on UD training data, starting from pretrained models. This script was used to generate `LAMBO` models.
- `run_training_splitting.py` -- train LAMBO subword splitting models on UD training data. 
- `run_tuning.py` -- tune existing LAMBO model to fit new data.
- `run_evaluation.py` -- evaluate existing models using UD gold standard.

Note that you can also extend LAMBO by modifying the data files that specify string that will be treated specially:
- [`emoji.tab`](src/lambo/resources/emoji.tab) includes a list of emojis (they will always be treated as separate tokens),
- [`pauses.txt`](src/lambo/resources/pauses.txt) include a list of verbal pauses (they will also be separated, but not split),
- [`turn_regexp.txt`](src/lambo/resources/turn_regexp.txt) enumerates regular expressions used to split turns (such as double newline),


## Credits

If you use LAMBO in your research, please cite it as software:
```bibtex
@software{LAMBO,
  author = {{Przyby{\l}a, Piotr}},
  title = {LAMBO: Layered Approach to Multi-level BOundary identification},
  url = {https://gitlab.clarin-pl.eu/syntactic-tools/lambo},
  version = {2.3},
  year = {2022},
}
```


## License

This project is licensed under the GNU General Public License v3.0.
