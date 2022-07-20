import warnings; warnings.filterwarnings("ignore")
import pickle
import time
import nltk
assert (nltk.__version__== '3.2.4')
from nltk.translate.bleu_score import sentence_bleu

file_name = '../models/IBM_Model1_table_50k_10it.pickle'  #Model file
text_file = "../dataset/ita.txt" #Dataset file

import_start_time = time.time()
print("[IBMModel1_50k3it] Loading models...")
with open(file_name, 'rb') as handle:
    table = pickle.load(handle)
print("[IBMModel1_50k3it] Model loaded in ", time.time()-import_start_time, " seconds")
perplexity_history = []

def clean(word):
    clean_word = word.lower()
    clean_word = clean_word.replace(".","")
    clean_word = clean_word.replace("!","")
    clean_word = clean_word.replace("'","")
    clean_word = clean_word.replace("\"","")
    clean_word = clean_word.replace(",","")
    clean_word = clean_word.replace("?","")
    return clean_word

def translate_eng_to_ita(table, sentence):
    result = list()

    for token in clean(sentence).split(" "):
        max_val = 0
        max_word = ""
        for (i,(eng_word,ita_word)) in enumerate(table):
            if eng_word==token:
                if table[eng_word, ita_word] > max_val:
                    max_val = table[eng_word, ita_word]
                    max_word = ita_word
        result.append(max_word)

    return " ".join(result)

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

def metric(target,output):
    import warnings
    warnings.filterwarnings("ignore")
    reference = [target.split(" ")]
    candidate = output.split(" ")
    score = sentence_bleu(reference, candidate)
    return score

num_of_test = 30
mean = 0
for i in range(num_of_test):
    sentence = test_pairs[i]
    eng_sentence = clean(sentence[0])
    ita_sentence = clean(sentence[1])
    translated = clean(translate_eng_to_ita(table, eng_sentence))
    m = metric(ita_sentence, translated)

    print("=================== TEST #", i, "===================")
    print("🇮🇹 ", ita_sentence)
    print("🇺🇸 ", eng_sentence)
    print("Translated: ", translated)
    print("BLEU score: ", m)
    mean += m
print("\nResult:", (mean / num_of_test))