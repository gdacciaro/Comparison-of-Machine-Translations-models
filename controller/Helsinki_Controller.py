import warnings

warnings.filterwarnings("ignore")

import warnings

from transformers import AutoModelForSeq2SeqLM
import torch # !! Required !!
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import sentence_bleu
import nltk
assert (nltk.__version__== '3.2.4')

from transformers import AutoTokenizer

it_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
it_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")


def helsinki_translate(sentence):
    encoded_input = it_en_tokenizer(sentence, return_tensors="pt")
    output = it_en_model.generate(**encoded_input)
    translated = it_en_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return translated