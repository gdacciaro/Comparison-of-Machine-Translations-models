import warnings
import time
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
import nltk
assert (nltk.__version__== '3.2.4')

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

def clean(word):
    clean_word = word.lower()
    clean_word = clean_word.replace(".","")
    #clean_word = clean_word.replace("!","")
    #clean_word = clean_word.replace("'","")
    clean_word = clean_word.replace("\"","")
    clean_word = clean_word.replace(",","")
    #clean_word = clean_word.replace("?","")
    clean_word = clean_word.replace("[start]","")
    clean_word = clean_word.replace("[end]","")
    clean_word = clean_word.strip()
    return clean_word


def metric(target,output):
    import warnings
    warnings.filterwarnings("ignore")
    reference = [target.split(" ")]
    candidate = output.split(" ")
    score = sentence_bleu(reference, candidate)
    return score

def decode_sequence(model, sentence):
    inputs = tokenizer(prefix + sentence, return_tensors="tf").input_ids
    outputs = model.generate(inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def translator(sentence):
    start_time = time.time()
    print("Translating: "+sentence)
    translated = decode_sequence(model_loaded, sentence)
    print("Translating: "+sentence, " -> ", translated, " in ", time.time()-start_time, " seconds")
    return translated


text_file = "../dataset/ita.txt"

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, ita, _ = line.split("\t")
    text_pairs.append((eng, ita))

num_train_samples = int(len(text_pairs) * 0.80)
train_pairs = text_pairs[:num_train_samples]
test_pairs = text_pairs[num_train_samples:]
print("")
print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(test_pairs)} test pairs")

num_of_test = 30
mean = 0
for i in range(num_of_test):
    sentence = test_pairs[i]
    eng_sentence = clean(sentence[0])
    ita_sentence = clean(sentence[1])
    translated = clean(translator(eng_sentence))
    m = metric(ita_sentence, translated)

    print("=================== TEST #", i, "===================")
    print("🇮🇹 ", ita_sentence)
    print("🇺🇸 ", eng_sentence)
    print("Translated: ", translated)
    print("BLEU score: ", m)
    mean += m
print("\n\n")
print("Result:", (mean / num_of_test))
