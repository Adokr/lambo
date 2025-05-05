"""
Short demo on using COMBO with LAMBO as input segmenter
"""
from combo.predict import COMBO
from combo.utils import lambo

if __name__=='__main__':
	# Load COMBO models
	nlp_new = COMBO.from_pretrained("polish-herbert-base-ud29",tokenizer=lambo.LamboTokenizer())
	
	# Running with pre-specified sentence boundaries
	text = ["To zdanie jest OK 👍🏿.", "To jest drugie zdanie."]
	sentences = nlp_new(text)
	print("{:5} {:15} {:15} {:10} {:10} {:10}".format('ID', 'TOKEN', 'LEMMA', 'UPOS', 'HEAD', 'DEPREL'))
	for sentence in sentences:
		for token in sentence.tokens:
			print("{:5} {:15} {:15} {:10} {:10} {:10}".format(str(token.id), token.token, token.lemma, token.upostag, str(token.head), token.deprel))
		print("\n")
	
	# Letting LAMBO decide on sentence boundaries
	text="To zdanie jest OK 👍🏿. To jest drugie zdanie."
	sentences = nlp_new(text)
	print("{:5} {:15} {:15} {:10} {:10} {:10}".format('ID', 'TOKEN', 'LEMMA', 'UPOS', 'HEAD', 'DEPREL'))
	for sentence in sentences:
		for token in sentence.tokens:
			print("{:5} {:15} {:15} {:10} {:10} {:10}".format(str(token.id), token.token, token.lemma, token.upostag, str(token.head), token.deprel))
		print("\n")
