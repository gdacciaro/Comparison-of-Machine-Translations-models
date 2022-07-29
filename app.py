#HLT Project 2021/2022, Master Degree Computer Science - University of Pisa

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import time

from controller.download_weights_if_necessary import download_weights_if_necessary
download_weights_if_necessary()

print("[SERVER] Server loading...")
import_start_time = time.time()

from controller.FS_Transformers_Controller import fs_transformers_translate
from controller.T5_Encoder_Decoder_Scratch_Controller import encT5decScratch_translate
from controller.FS_GRU_Controller import fs_gru_translate
from controller.DeepL_Controller import deepl_translate
from controller.Helsinki_Controller import helsinki_translate
from controller.IBMModel_Controller import ibm_translate
from controller.T5_Controller import t5_translate

app = Flask(__name__, static_url_path='/static')
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

print("[SERVER] Server loaded in ", time.time()-import_start_time, " seconds")

@cross_origin()
@app.route('/translate')
def query_example():
    model = request.args.get('model')
    sentence = request.args.get('sentence')

    if model == 'IBM Model 1 (50k)' or model == 'IBM Model 1':
        return jsonify({"response": ibm_translate(sentence)})
    if model == 'GRU (Custom)':
        return jsonify({"response": fs_gru_translate(sentence)})
    if model == 'Transformer (Custom)':
        return jsonify({"response": fs_transformers_translate(sentence)})
    if model == 'T5':
        return jsonify({"response": t5_translate(sentence)})
    #if model == 'Bert2Bert':
    #    return jsonify({"response": fs_lstm_translate(sentence)})
    if model == 'encT5/decScratch':
        return jsonify({"response": encT5decScratch_translate(sentence)})
    if model == 'DeepL':
        return jsonify({"response": deepl_translate(sentence)})
    if model == 'Helsinki':
        return jsonify({"response": helsinki_translate(sentence)})

    return jsonify({"response": "error"})


if __name__ == '__main__':
    app.run()
