import warnings
import time
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

warnings.filterwarnings("ignore")

from nltk.translate.bleu_score import sentence_bleu
import nltk
#assert (nltk.__version__== '3.2.4')

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


def translator(sentence):
    import requests
    deepl_token = "d014baae-80fc-29d6-ec7f-fd7e7d8b8fb5:fx"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = 'auth_key=' + deepl_token + '&text=' + sentence + '&target_lang=IT'

    response = requests.post('https://api-free.deepl.com/v2/translate', headers=headers, data=data)
    return response.json()['translations'][0]["text"]

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
