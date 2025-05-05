"""
Rough procedure to generate languages.txt from a UD folder. Includes all languages that have a test and dev portions.
Uses the previous version of the file to translate language names to ISO codes. Selects the largest corpus as preferred
 for the language. May require manual adjustment to exclude abnormal treebanks or add missing
 ISO codes for new languages.
 Manual changes in version 2.13:
- excluding UD_Arabic-NYUAD
- adding '?' as ISO code for the new languages outside the standard
- selected UD_German-GSD as default in place of UD_German-HDT, which lacks spacing information
- corrected language code for UD_Norwegian-Bokmaal from nn to no
- added UD_Polish-NKJP1M_PDB from https://huggingface.co/datasets/ipipan/nlprepl/tree/main/ud_tagset/fair_by_document_type/_conllu
"""
from pathlib import Path

old_languages_txt = '/Users/piotr/projects/lambo/src/lambo/resources/languages.txt'
new_ud_treebanks = '/Users/piotr/data/ud-treebanks-v2.13'

codedict = {}
for line in open(old_languages_txt):
    if line.startswith('#'):
        continue
    parts = line.strip().split(' ')
    lang = parts[2]
    code = parts[1]
    codedict[lang] = code

udpath = Path(new_ud_treebanks)

subdirs = [x for x in udpath.iterdir() if x.is_dir()]
subdirs.sort()

sizes = {}

for subdir in subdirs:
    hasTrain = False
    hasDev = False
    hasTest = False
    trainfile = None
    for file in subdir.iterdir():
        if file.name.endswith('train.txt'):
            hasTrain = True
            trainfile = file
        elif file.name.endswith('test.txt'):
            hasTest = True
        elif file.name.endswith('dev.txt'):
            hasDev = True
    if (not hasTrain) or (not hasTest) or (not hasDev):
        continue
    treebank_name = subdir.name
    language_name = treebank_name[3:].split('-')[0]
    code = '@@@@@'
    if language_name in codedict:
        code = codedict[language_name]
    if language_name not in sizes:
        sizes[language_name] = {}
    sizes[language_name][treebank_name] = trainfile.stat().st_size

for language_name in sizes:
    maxlen = 0
    best = None
    for treebank_name in sizes[language_name]:
        if sizes[language_name][treebank_name] > maxlen:
            best = treebank_name
            maxlen = sizes[language_name][treebank_name]
    if len(sizes[language_name]) > 1:
        sizes[language_name]['preferred'] = best

print(
    "# Format: <UD training corpus> <ISO 639-1 code (for OSCAR pretraining)> <Language name> <Recommended (chosen by size)>")
for subdir in subdirs:
    hasTrain = False
    hasDev = False
    hasTest = False
    trainfile = None
    for file in subdir.iterdir():
        if file.name.endswith('train.txt'):
            hasTrain = True
            trainfile = file
        elif file.name.endswith('test.txt'):
            hasTest = True
        elif file.name.endswith('dev.txt'):
            hasDev = True
    if (not hasTrain) or (not hasTest) or (not hasDev):
        continue
    treebank_name = subdir.name
    language_name = treebank_name[3:].split('-')[0]
    code = '@@@@@'
    if language_name in codedict:
        code = codedict[language_name]
    preferred = ''
    if 'preferred' in sizes[language_name] and sizes[language_name]['preferred'] == treebank_name:
        preferred = ' *'
    print(treebank_name + ' ' + code + ' ' + language_name + preferred)

