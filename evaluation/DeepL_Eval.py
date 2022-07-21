import warnings
from evaluation.evalutator import evaluate_model
warnings.filterwarnings("ignore")


def translate(sentence):
    import requests
    deepl_token = "d014baae-80fc-29d6-ec7f-fd7e7d8b8fb5:fx"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = 'auth_key=' + deepl_token + '&text=' + sentence + '&target_lang=IT'

    response = requests.post('https://api-free.deepl.com/v2/translate', headers=headers, data=data)
    return response.json()['translations'][0]["text"]


result = evaluate_model(translate)
print("Result:",result)