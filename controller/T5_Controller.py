import warnings
import time
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

print("[T5] Loading models...")
import_start_time = time.time()

model_checkpoint = "t5-small" # Load the model
model_loaded = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="../.cache")
model_loaded.load_weights('models/t5_weights/') #Don't put "../" in the path!
model_loaded.summary()

print("[T5] Model loaded in ", time.time()-import_start_time, " seconds")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, cache_dir="../.cache")

prefix = "translate English to Italian: "

def decode_sequence(model, sentence):
    inputs = tokenizer(prefix + sentence, return_tensors="tf").input_ids
    outputs = model.generate(inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def t5_translate(sentence):
    start_time = time.time()
    print("Translating: "+sentence)
    translated = decode_sequence(model_loaded, sentence)
    print("Translating: "+sentence, " -> ", translated, " in ", time.time()-start_time, " seconds")
    return translated