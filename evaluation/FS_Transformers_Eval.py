import re
import string

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.layers import TextVectorization

vocab_size = 15000
sequence_length = 20
max_decoded_sentence_length = 20

text_file = "../dataset/ita.txt"

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, ita, _ = line.split("\t")
    ita = "[start] " + ita + " [end]"
    text_pairs.append((eng, ita))

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,)
ita_vectorization = TextVectorization(max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length+1, standardize=custom_standardization,)

train_eng_texts = [pair[0] for pair in train_pairs]
train_ita_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
ita_vectorization.adapt(train_ita_texts)

ita_vocab = ita_vectorization.get_vocabulary()
ita_index_lookup = dict(zip(range(len(ita_vocab)), ita_vocab))

def decode_sequence(model,input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = ita_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = ita_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


model_loaded = tf.saved_model.load("../models/fs_transformer_weights/")

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

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, ita, _ = line.split("\t")
    ita = "[start] " + ita + " [end]"
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
    translated = clean(decode_sequence(model_loaded, eng_sentence))
    m = metric(ita_sentence, translated)

    print("=================== TEST #", i, "===================")
    print("🇮🇹 ", ita_sentence)
    print("🇺🇸 ", eng_sentence)
    print("Translated: ", translated)
    print("BLEU score: ", m)
    mean += m
print("\nResult:", (mean / num_of_test))