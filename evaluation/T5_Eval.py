import warnings
import time
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.evalutator import evaluate_model

warnings.filterwarnings("ignore")
import_start_time = time.time()
print("[T5] Loading models...")
model_checkpoint = "t5-small"  ## "t5_weights-small", "t5_weights-base", "t5_weights-larg", "t5_weights-3b", "t5_weights-11b"
model_loaded = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="../.cache")
model_loaded.load_weights('../models/t5_weights/')
model_loaded.summary()
print("[T5] Model loaded in ", time.time()-import_start_time, " seconds")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, cache_dir="../.cache")

prefix = "translate English to Italian: "

def translate(sentence):
    inputs = tokenizer(prefix + sentence, return_tensors="tf").input_ids
    outputs = model_loaded.generate(inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

result = evaluate_model(translate)
print("Result:",result)
