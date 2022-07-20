#Inspired by : https://www.kaggle.com/code/garbo999/machine-translation-ibm-model-1-em-algorithm/notebook

import warnings; warnings.filterwarnings("ignore")
import pickle
import time

file_name = 'models/IBM_Model1_table_50k_3it.pickle' # Don't put "../" in the path!
import_start_time = time.time()
print("[IBMModel1_50k3it] Loading models...")
with open(file_name, 'rb') as handle:
    table = pickle.load(handle)
print("[IBMModel1_50k3it] Model loaded in ", time.time()-import_start_time, " seconds")
perplexity_history = []

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

def translate_eng_to_ita(table, sentence):
    result = list()

    for token in clean(sentence).split(" "):
        max_val = 0
        max_word = "<not_found>"
        for (i,(eng_word,ita_word)) in enumerate(table):
            if eng_word==token:
                if table[eng_word, ita_word] > max_val:
                    max_val = table[eng_word, ita_word]
                    max_word = ita_word
        result.append(max_word)

    return " ".join(result)


def ibm_translate(sentence):
    start_time = time.time()
    print("Translating: "+sentence)
    translated =  translate_eng_to_ita(table, sentence)
    print("Translating: "+sentence, " -> ", translated, " in ", time.time()-start_time, " seconds")
    return translated