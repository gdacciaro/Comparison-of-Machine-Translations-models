import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFBertModel

text_file = "../dataset/ita.txt"

sequence_length = 90

def split_set(dataset: tf.data.Dataset,
              tr: float = 0.8,
              val: float = 0.1,
              ts: float = 0.1,
              shuffle: bool = True) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    if tr + val + ts != 1:
        raise ValueError("Train, validation and test partition not allowed with such splits")

    dataset_size = dataset.cardinality().numpy()
    if shuffle:
        dataset = dataset.shuffle(dataset_size)

    tr_size = int(tr * dataset_size)
    val_size = int(val * dataset_size)

    tr_set = dataset.take(tr_size)
    val_set = dataset.skip(tr_size).take(val_size)
    ts_set = dataset.skip(tr_size).skip(val_size)
    return tr_set, val_set, ts_set


def make_batches(dataset_src_dst: tf.data.Dataset, batch_size: int):
    return dataset_src_dst.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def create_dataset(name: str, preprocessed: bool):
    with open(name, encoding="UTF-8") as datafile:
        src_set = list()
        dst_set = list()
        for sentence in datafile:
            sentence = sentence.split("\t")
            src_set.append(sentence[0])
            if preprocessed:
                dst_set.append(sentence[1].split("\n")[0])
            else:
                dst_set.append(sentence[1])

    return src_set, dst_set


def format_dataset(src, trg):
    return ({"encoder_inputs": src, "decoder_inputs": trg[:, :-1]}, trg[:, 1:])


def make_dataset(dataset, batch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()


"""# **Tokenizer**

"""

from transformers import BertTokenizerFast

source_src = "bert-base-uncased"
target_src = "dbmdz/bert-base-italian-cased"

tokenizer_src = BertTokenizerFast.from_pretrained(source_src, cache_dir ="../cache")
tokenizer_trg = BertTokenizerFast.from_pretrained(target_src, cache_dir ="../cache")


def tokenize(entire_set):
    source_set, target_set = entire_set
    tokens_source = tokenizer_src(source_set, truncation=True, padding="max_length",
                                  return_tensors="tf", max_length=sequence_length).data["input_ids"]
    tokens_source = tf.cast(tokens_source, dtype=tf.int32)
    tokens_target = tokenizer_trg(target_set, add_special_tokens=True,
                                  truncation=True, padding="max_length",
                                  return_tensors="tf", max_length=sequence_length + 1).data["input_ids"]
    tokens_target = tf.cast(tokens_target, dtype=tf.int32)
    return tokens_source, tokens_target


v_size_src = tokenizer_src.vocab_size
v_size_trg = tokenizer_trg.vocab_size

dataset = tf.data.Dataset.from_tensor_slices(tokenize(create_dataset("dataset/ita.txt", False)))

tr_set, val_set, ts_set = split_set(dataset, 0.9, 0.05, 0.05)

batch_size = 16
train_ds = make_dataset(tr_set, batch_size)
val_ds = make_dataset(val_set, batch_size)

"""# **Encoder-Decoder Model**"""

dropout_rate = 0.2

encoder = TFBertModel.from_pretrained(source_src, cache_dir ="../cache")


class EncoderLayer(layers.Layer):

    def __init__(self, layers_size: int, dense_size: int, num_heads: int, dropout=0.1, **kwargs) -> None:
        super(EncoderLayer, self).__init__(**kwargs)

        self.layers_size = layers_size
        self.dense_size = dense_size
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_size, activation="relu"), layers.Dropout(dropout), layers.Dense(layers_size)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        else:
            print("Mask not built")
            assert False

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class EncoderTransformer(layers.Layer):

    def __init__(self,
                 num_layers: int,
                 layers_size: int,
                 dense_size: int,
                 num_heads: int,
                 max_length: int,
                 v_size_src: int,
                 dropout: float = 0.1) -> None:
        super(EncoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(max_length, v_size_src, layers_size)
        self.enc_layers = [EncoderLayer(layers_size, dense_size, num_heads) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        src_embeddings = self.pos_embedding(inputs)
        enc_out = self.dropout(src_embeddings)
        for i in range(self.num_layers):
            enc_out = self.enc_layers[i](enc_out)

        return enc_out  # (batch_size, input_seq_len, layers_size)


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


class DecoderLayer(layers.Layer):

    def __init__(self, layers_size: int, dense_size: int, num_heads: int, dropout=0.1, **kwargs) -> None:
        super(DecoderLayer, self).__init__(**kwargs)

        self.layers_size = layers_size
        self.dense_size = dense_size
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.attention_2 = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_size, activation="relu"), layers.Dropout(dropout), layers.Dense(layers_size)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, encoder_outputs: tf.Tensor, mask=None) -> tf.Tensor:
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

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


class DecoderTransformer(layers.Layer):

    def __init__(self,
                 num_layers: int,
                 layers_size: int,
                 dense_size: int,
                 num_heads: int,
                 max_length: int,
                 v_size_dst: int,
                 dropout=0.1) -> None:
        super(DecoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(max_length, v_size_dst, layers_size)
        self.dec_layers = [DecoderLayer(layers_size, dense_size, num_heads) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, enc_output: tf.Tensor, mask=None) -> tf.Tensor:
        dst_embeddings = self.pos_embedding(inputs)
        dec_output = self.dropout(dst_embeddings)
        for i in range(self.num_layers):
            dec_output = self.dec_layers[i](dec_output, enc_output)

        return dec_output


sequence_length = 90  # ?


def create_model(layers_size: int, num_layers: int, dense_size: int, num_heads: int, max_length: int,
                 encoder=None) -> tf.keras.Model:
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    if encoder is not None:
        outputs = encoder(encoder_inputs)
        encoder_outputs = outputs.last_hidden_state
        layers_size = encoder_outputs.shape[-1]  # the size of the encoder and decoder layers must be the same
    else:
        encoder_outputs = EncoderTransformer(num_layers, layers_size, dense_size, num_heads, max_length, v_size_src)(
            encoder_inputs)

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = tf.keras.Input(shape=(None, layers_size), name="decoder_state_inputs")
    decoder_outputs = DecoderTransformer(num_layers, layers_size, dense_size, num_heads, max_length, v_size_trg)(decoder_inputs, encoded_seq_inputs)
    decoder_outputs = layers.Dense(v_size_trg, activation="softmax")(decoder_outputs)
    decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    # Final models
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
    return transformer


d_model = 512

model = create_model(512, 7, 2048, 8, 80, encoder)
model.load_weights('./save')
model.summary()
'''
opt = tf.keras.optimizers.Adam()
train_ds = train_ds.shuffle(10 ** 6)
models.summary()
models.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
models.fit(train_ds, epochs=1, validation_data=val_ds, shuffle=True)
'''

