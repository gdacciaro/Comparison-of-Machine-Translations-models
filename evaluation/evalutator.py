text_file = "../dataset/ita.txt"
num_of_tests = 100
import time
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, ita, _ = line.split("\t")
    ita = "[start] " + ita + " [end]"
    text_pairs.append((eng, ita))

import random
random.seed(39081)
random.shuffle(text_pairs)

num_train_samples = int(len(text_pairs) * 0.80)
train_pairs = text_pairs[:num_train_samples]
test_pairs = text_pairs[num_train_samples:]

def the_metric(target, output):
    import evaluate
    predictions = [output]
    references = [target]
    bleu = evaluate.load("bleu")
    try:
        results = bleu.compute(predictions=predictions, references=references)["bleu"]
    except:
        results = 0
    return results

def clean(word):
    clean_word = word.lower()
    clean_word = clean_word.replace(".", "")
    clean_word = clean_word.replace("\"", "")
    clean_word = clean_word.replace(",", "")
    clean_word = clean_word.replace("’", "'")
    clean_word = clean_word.replace("[start]", "")
    clean_word = clean_word.replace("[end]", "")
    clean_word = clean_word.strip()
    return clean_word


def evaluate_model(translate_function):
    result = 0
    count = 0
    total_time = 0
    for i, item in enumerate(test_pairs[0:num_of_tests]):
        start_time = time.time()
        input_eng = clean(item[0])  # eng
        target = clean(item[1])  # ita
        output = clean(translate_function(input_eng))
        end_time = time.time()
        total_time += end_time - start_time
        item_result = the_metric(target, output)
        print("=================== TEST #", i, "===================")
        print("🇮🇹 ", target)
        print("🇺🇸 ", input_eng)
        print("Translated: ", output)
        print("BLEU score: ", item_result)
        result += item_result
        count += 1
    print("\n\n=================== TEST RESULT ===================")
    print("Average BLEU score: ", result / count)
    print("Total inference time: ",total_time)
    print("Average inference time: ",total_time/ count)
    print("=================== TEST RESULT ===================")
    final_result = result / count
    return final_result