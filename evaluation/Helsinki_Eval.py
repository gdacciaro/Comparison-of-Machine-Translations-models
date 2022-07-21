import warnings
import time
from transformers import AutoModelForSeq2SeqLM
import torch # !! Required !!

from evaluation.evalutator import evaluate_model

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

print("[Helsinki] Loading model...")
import_start_time = time.time()
it_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
it_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
print("[Helsinki] Model loaded in ", time.time()-import_start_time, " seconds")

def translate(input):
    encoded_input = it_en_tokenizer(input, return_tensors="pt")
    output = it_en_model.generate(**encoded_input)
    translated = it_en_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return translated

result = evaluate_model(translate)
print("Result:",result)
