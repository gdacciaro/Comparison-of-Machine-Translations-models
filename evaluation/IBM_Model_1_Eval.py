import warnings; from evaluation.evalutator import evaluate_model, clean

import pickle
import time

file_name = '../models/IBM_Model1_table_50k_3it.pickle'  #Model file
text_file = "../dataset/ita.txt" #Dataset file

import_start_time = time.time()
print("[IBMModel1_50k3it] Loading models...")
with open(file_name, 'rb') as handle:
    table = pickle.load(handle)
print("[IBMModel1_50k3it] Model loaded in ", time.time()-import_start_time, " seconds")

def translate(sentence):
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

result = evaluate_model(translate)
print("Result:",result)