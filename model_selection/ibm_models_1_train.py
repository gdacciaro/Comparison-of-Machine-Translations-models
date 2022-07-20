#Inspired by : https://www.kaggle.com/code/garbo999/machine-translation-ibm-model-1-em-algorithm/notebook

import warnings; warnings.filterwarnings("ignore")
import random
import time
import numpy as np
import pickle
from nltk.translate.bleu_score import sentence_bleu

sentence_used_from_dataset = 100000
num_iterations = 3
num_of_test = 10
perplexity_history = []
debug = True
s_total = {}
epsilon = 1
table = {}
file_name = 'IBM_Model1_table.pickle'

#Constants
english_index = 0
italian_index = 1

#Telegram API
token = "5597879510:AAH1FSuZa7lA_xoAp-7JiUDE0OY38p-Tq5M"
theUrl = "https://api.telegram.org/bot"+token+"/sendMessage"
who = "-606080513"

def sendMessage(text=""):
    import json
    import requests
    data = {'chat_id': who, 'disable_notification': 'false', 'text': text}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    requests.post(theUrl, data=json.dumps(data, ensure_ascii=True), headers=headers)
    print(data)

def sendFile(file):
    import telepot
    theFile = open(file, 'rb')
    bot = telepot.Bot(token)
    bot.sendDocument(who, theFile)

text_file = "dataset/ita.txt"

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, ita, _ = line.split("\t")
    text_pairs.append((eng, ita))
    
random.shuffle(text_pairs)
num_train_samples = int(len(text_pairs)*0.80)
train_pairs = text_pairs[:num_train_samples ]
test_pairs = text_pairs[num_train_samples:]
print("")
print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(test_pairs)} test pairs")

def clean(word):
    clean_word = word.lower()
    clean_word = clean_word.replace(".","")
    clean_word = clean_word.replace("!","")
    clean_word = clean_word.replace("'","")
    clean_word = clean_word.replace("\"","")
    clean_word = clean_word.replace(",","")
    clean_word = clean_word.replace("?","")
    return clean_word

sentence_pairs = []
italian_words = []
english_words = []

for i in range(sentence_used_from_dataset):
    sentence = train_pairs[i]
    eng_sentence = clean(sentence[english_index])
    ita_sentence = clean(sentence[italian_index])
    sentence_pairs.append([ eng_sentence.split(" ") , ita_sentence.split(" ")])
    for word_ita in ita_sentence.split(" "):
        italian_words.append(word_ita)
    for word_eng in eng_sentence.split(" "):
        english_words.append(word_eng)

english_words = sorted(list(set(english_words)), key=lambda s: s.lower())
italian_words = sorted(list(set(italian_words)), key=lambda s: s.lower())

english_vocab_size = len(english_words)
italian_vocab_size = len(italian_words)

print('english_vocab_size: ', english_vocab_size)
print('italian_vocab_size: ', italian_vocab_size)

"""# Statistical Machine Translation"""

# Input: english sentence e, foreign sentence f, hash of translation probabilities t, epsilon 
# Output: probability of e given f

def probability_e_f(english_words, italian_words, table, epsilon=1):
    english_size = len(english_words)
    italian_size = len(italian_words)
    p_e_f = 1
    
    for english_word in english_words: # iterate over english words ew in english sentence e
        inner_sum = 0
        for italian_word in italian_words: # iterate over foreign words fw in foreign sentence f
            inner_sum += table[(english_word, italian_word)]
        p_e_f = inner_sum * p_e_f
    
    p_e_f = p_e_f * epsilon / (italian_size**english_size)
    
    return p_e_f

# Input: Collection of sentence pairs sentence_pairs, hash of translation probabilities t, epsilon
# Output: Perplexity of model

def perplexity(sentence_pairs, t, epsilon=1):
    result = 0
    for pair in sentence_pairs:
        prob = probability_e_f(english_words=pair[english_index],
                               italian_words=pair[italian_index],
                               table=t,
                               epsilon=epsilon)
        result += np.log2(prob)
    return np.power(2,-result)

def init_prob(table, init_val, english_words, italian_words):
    for word_ita in italian_words:
        for word_eng in english_words:
            tup = (word_eng, word_ita) # tuple required because dict key cannot be list
            table[tup] = init_val

""" Main routine """
start_time = time.time()

# Initialize probabilities uniformly
init_val = 1.0 / italian_vocab_size
init_prob(table, init_val, english_words, italian_words)

for iter in range(num_iterations):
    print("Iteration #",iter)
    # Calculate perplexity
    pp = perplexity(sentence_pairs, table, epsilon)
    perplexity_history.append(pp)

    # Initialize
    count = {}
    total = {}

    for word_ita in italian_words:
        total[word_ita] = 0.0
        for word_eng in english_words:
            count[(word_eng, word_ita)] = 0.0

    for sp in sentence_pairs:
        # Compute normalization
        for word_eng in sp[english_index]:
            s_total[word_eng] = 0.0
            for word_ita in sp[italian_index]:
                s_total[word_eng] += table[(word_eng, word_ita)]

        # Collect counts
        for word_eng in sp[english_index]:
            for word_ita in sp[italian_index]:
                count[(word_eng, word_ita)] += table[(word_eng, word_ita)] / s_total[word_eng]
                total[word_ita] += table[(word_eng, word_ita)] / s_total[word_eng]

    # Estimate probabilities
    for word_ita in italian_words:
        for word_eng in english_words:
            table[(word_eng, word_ita)] = count[(word_eng, word_ita)] / total[word_ita]

end_time = time.time()

# -------------------------------------------------------------------------------------------------------------- #
#Save the table

with open(file_name, 'wb') as handle:
    pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------------------------------- #

def translate_eng_to_ita(sentence):
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

def translate_ita_to_eng(sentence):
    result = list()

    for token in clean(sentence).split(" "):
        max_val = 0
        max_word = "<not_found>"
        for (i,(eng_word,ita_word)) in enumerate(table):
            if ita_word==token:
                if table[eng_word, ita_word] > max_val:
                    max_val = table[eng_word, ita_word]
                    max_word = eng_word
        result.append(max_word)

    return " ".join(result)

"""# Metric"""

def metric(target,output):
    import warnings
    warnings.filterwarnings("ignore")
    reference = [target.split(" ")]
    candidate = output.split(" ")
    score = sentence_bleu(reference, candidate)
    return score

"""# Evalutation"""

mean = 0

for i in range(num_of_test):
    sentence = random.choice(test_pairs)
    eng_sentence = clean(sentence[english_index])
    ita_sentence = clean(sentence[italian_index])
    translated = translate_eng_to_ita(eng_sentence)
    m = metric(ita_sentence,translated)
    
    if debug: 
        print("=================== TEST #", i, "===================")
        print("🇮🇹 ",ita_sentence)
        print("🇺🇸 ",eng_sentence)
        print("Translated: ",translated)
        print("BLEU score: ", m)
    mean += m

print("Result:", (mean / num_of_test))

sendMessage("============\nIBM Model 1 result\nmetric:BLEU\nnum_of_trial: "+str(num_of_test)
            +"\nresult:"+str((mean / num_of_test))
            +"\npseudo-training time:"+str(round(end_time-start_time,6))
            +"\n============")

