import warnings
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import T5Tokenizer, BertTokenizer, TFT5EncoderModel

from evaluation.evalutator import evaluate_model


warnings.filterwarnings("ignore")


# Detect hardware
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
except ValueError:
    tpu_resolver = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

# Select appropriate distribution strategy
if tpu_resolver:
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
    print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU')

print("Number of accelerators: ", strategy.num_replicas_in_sync)

sequence_length = 90
dropout_rate = 0.2


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="elu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        else:
            assert False
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="elu"), layers.Dropout(dropout_rate), layers.Dense(embed_dim)])
        # self.dense_proj_f = keras.Sequential(
        #    [layers.Dense(latent_dim, activation="elu"), layers.Dropout(dropout_rate), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        # self.layernorm_1_f = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        # self.dropout_1 = layers.Dropout(dropout_rate)
        # self.dropout_2 = layers.Dropout(dropout_rate)

        self.resid1 = layers.Add()
        self.resid2 = layers.Add()
        self.resid3 = layers.Add()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )

        # attention_output_1 = self.dropout_1(attention_output_1)
        out_1 = self.layernorm_1(self.resid1([inputs, attention_output_1]))
        # proj_output_f = self.dense_proj_f(out_1)
        # out_1 = self.layernorm_1_f(layers.Add()([out_1, proj_output_f]))

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask
        )
        # attention_output_2 = self.dropout_2(attention_output_2)
        out_2 = self.layernorm_2(self.resid2([out_1, attention_output_2]))

        proj_output = self.dense_proj(out_2)
        # proj_output = self.dropout_1(proj_output)
        return self.layernorm_3(self.resid3([out_2, proj_output]))

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


def create_model(encoder, embed_dim, v_size_trg, latent_dim, num_heads):
  encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
  outputs = encoder(encoder_inputs)
  encoder_outputs = outputs.last_hidden_state

  decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
  encoded_seq_inputs = tf.keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
  x = PositionalEmbedding(sequence_length, v_size_trg, embed_dim)(decoder_inputs)
  x = layers.Dropout(dropout_rate)(x)
  for i in range(8):
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)

  decoder_outputs = layers.Dense(v_size_trg, activation="softmax")(x)

  decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

  decoder_outputs = decoder([decoder_inputs, encoder_outputs])
  transformer = tf.keras.Model(
      [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
  )
  return transformer


import_start_time = time.time()
print("[T5 Encoder - Decoder from Scratch] Loading models...")

embed_dim = 512
latent_dim = 1024
num_heads = 8

source_src= "google/t5-v1_1-small"
target_src = "dbmdz/bert-base-italian-cased"

tokenizer_src = T5Tokenizer.from_pretrained(source_src)
tokenizer_trg = BertTokenizer.from_pretrained(target_src)

v_size_src = tokenizer_src.vocab_size
v_size_trg = tokenizer_trg.vocab_size

encoder = TFT5EncoderModel.from_pretrained(source_src)
transformer = create_model(encoder, embed_dim, v_size_trg, latent_dim, num_heads)
transformer.load_weights("./trained_models/t5encoder-decoder.h5")
print("[T5 Encoder - Decoder from Scratch] Model loaded in ", time.time()-import_start_time, " seconds")


def decode_sequence(input_sentence, tokenizer_source, tokenizer_target, transformer):
    # tokenized_input_sentence=input_sentence
    tokenized_input_sentence = \
    tokenizer_source(input_sentence, return_tensors='tf', add_special_tokens=True, max_length=sequence_length,
                     padding='max_length', truncation=True).data["input_ids"]
    decoded_sentence = "[CLS]"
    list_tokens = [decoded_sentence]
    for i in range(sequence_length):

        decoded_sentence = tokenizer_target.convert_tokens_to_string(list_tokens)
        tokenized_target_sentence = \
        tokenizer_target(decoded_sentence, return_tensors='tf', add_special_tokens=False, max_length=sequence_length,
                         padding='max_length').data['input_ids']
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = tokenizer_target.ids_to_tokens[sampled_token_index]  # spa_index_lookup[sampled_token_index]

        # decoded_sentence += sampled_token

        if sampled_token == "[SEP]":
            decoded_sentence = tokenizer_target.convert_tokens_to_string(list_tokens[1:])
            break
        list_tokens.append(sampled_token)

    return list_tokens, decoded_sentence


def translate(sequence):
  with strategy.scope():
    tokens, translated = decode_sequence(sequence, tokenizer_src, tokenizer_trg, transformer)
  return translated


result = evaluate_model(translate)
print("Result:",result)