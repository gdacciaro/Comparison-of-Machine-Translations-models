import warnings
import time
from transformers import AutoModelForSeq2SeqLM
import torch # !! Required !!
warnings.filterwarnings("ignore")
from nltk.translate.bleu_score import sentence_bleu
import nltk
print("nltk.__version__:", nltk.__version__)
#assert (nltk.__version__== '3.2.4')

from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

print("[Helsinki] Loading model...")
import_start_time = time.time()
it_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-it")
it_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-it")
print("[Helsinki] Model loaded in ", time.time()-import_start_time, " seconds")

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

    encoded_input = it_en_tokenizer(eng_sentence, return_tensors="pt")
    output = it_en_model.generate(**encoded_input)
    translated = it_en_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    translated = clean(translated)
    m = metric(ita_sentence, translated)

    print("=================== TEST #", i, "===================")
    print("🇮🇹 ", ita_sentence)
    print("🇺🇸 ", eng_sentence)
    print("Translated: ", translated)
    print("BLEU score: ", m)
    mean += m
print("\nResult:", (mean / num_of_test))