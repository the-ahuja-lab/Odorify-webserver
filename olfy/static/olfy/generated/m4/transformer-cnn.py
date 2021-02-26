import sys
import os
import time
import pickle
import math
import configparser
import numpy as np
import csv
import h5py
import tarfile
import shutil
import math
import random

from rdkit import Chem
from rdkit.Chem import SaltRemover
from layers import PositionLayer, MaskLayerLeft, \
    MaskLayerRight, MaskLayerTriangular, \
    SelfLayer, LayerNormalization

version = 4

# config = configparser.ConfigParser()
# config.read(sys.argv[1])
# path = sys.argv[2]
# def getConfig(section, attribute, default=""):
#     try:
#         return config[section][attribute]
#     except:
#         return default

# TRAIN = getConfig("Task","train_mode")
# MODEL_FILE = getConfig("Task","model_file")
# TRAIN_FILE = getConfig("Task","train_data_file")
# APPLY_FILE = getConfig("Task","apply_data_file", "train.csv")
# RESULT_FILE = getConfig("Task","result_file", f"{path}/results.csv")
# NUM_EPOCHS = int(getConfig("Details","n_epochs", "100"))
# BATCH_SIZE = int(getConfig("Details","batch_size", "32"))
# SEED = int(getConfig("Details","seed", "657488"))
# CANONIZE = getConfig("Details","canonize")
# DEVICE = getConfig("Details","gpu")
# EARLY_STOPPING = float(getConfig("Details", "early-sopping", "0.9"))
# AVERAGING = int(getConfig("Details", "averaging", "5"))
# FIXED_LEARNING_RATE = getConfig("Details", "fixed-learning-rate", "False")
# RETRAIN = getConfig("Details", "retrain", "False")
# CHIRALITY = getConfig("Details", "chirality", "True")

# FIRST_LINE = getConfig("Details", "first-line", "True")
# if FIRST_LINE == "True":
#    FIRST_LINE = True
# else:
#    FIRST_LINE = False
path = sys.argv[1]
TRAIN = "False"
MODEL_FILE = f'{path}/model56.tar'
# TRAIN_FILE = getConfig("Task","train_data_file")
APPLY_FILE = f'{path}/input1.csv'
RESULT_FILE = f"{path}/results.csv"
NUM_EPOCHS = 100
BATCH_SIZE = 32
SEED = 657488
CANONIZE = "False"
DEVICE = 'gpu'
EARLY_STOPPING = 0.9
AVERAGING = 5
FIXED_LEARNING_RATE = "False"
RETRAIN = "False"
CHIRALITY = "True"
FIRST_LINE = "False"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

CONV_OFFSET = 20
N_HIDDEN = 512
N_HIDDEN_CNN = 512
EMBEDDING_SIZE = 64
KEY_SIZE = EMBEDDING_SIZE
n_block, n_self = 3, 10

# our vocabulary
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
g_chars = set(chars)
vocab_size = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
tf.logging.set_verbosity(tf.logging.ERROR)
K.set_session(tf.Session(config=config))

try:
  tf.random.set_random_seed(SEED)
except:
  print("not supported tf.random.set_random_seed(SEED)")

np.random.seed(SEED)

props = {}
canon_pairs = []


class suppress_stderr(object):
  def __init__(self):
    self.null_fds = [os.open(os.devnull, os.O_RDWR)]
    self.save_fds = [os.dup(2)]

  def __enter__(self):
    os.dup2(self.null_fds[0], 2)

  def __exit__(self, *_):
    os.dup2(self.save_fds[0], 2)
    for fd in self.null_fds + self.save_fds:
      os.close(fd)

def findBoundaries(DS):

  for prop in props:

    x = []
    for i in range(len(DS)):
      if DS[i][2][prop] == 1:
        x.append(DS[i][1][prop])

    hist = np.histogram(x)[0]
    if np.count_nonzero(hist) > 2:
      y_min = np.min(x)
      y_max = np.max(x)

      add = 0.01 * (y_max - y_min)
      y_max = y_max + add
      y_min = y_min - add

      print(props[prop][1], "regression:", y_min, "to", y_max, "scaling...")

      for i in range(len(DS)):
        if DS[i][2][prop] == 1:
          DS[i][1][prop] = 0.9 + 0.8 * \
              (DS[i][1][prop] - y_max) / (y_max - y_min)

      props[prop].extend(["regression", y_min, y_max])

    else:
      props[prop].extend(["classification"])


def analyzeDescrFile(fname):

  first_row = FIRST_LINE

  DS = []
  ind_mol = 0

  remover = SaltRemover.SaltRemover()

  line = 0
  for row in csv.reader(open(fname, "r")):

    # if line > 100: break
    #line = line + 1

    if first_row:
      first_row = False
      j = 0
      for i, prop in enumerate(row):
        if prop == "smiles":
          ind_mol = i
          continue
        props[j] = [i, prop]
        j = j + 1
      continue
    else:
      props[0] = [1, 'property']
      ind_mol = 0

    mol = row[ind_mol].strip()

    # remove molecules with symbols not in our vocabulary
    g_mol = set(mol)
    g_left = g_mol - g_chars
    if len(g_left) > 0:
      continue

    arr = []
    canon = ""

    try:
      if CANONIZE == 'True':
        with suppress_stderr():
          m = Chem.MolFromSmiles(mol)
          m = remover.StripMol(m)

          if m is not None:
            canon = Chem.MolToSmiles(m)

          if m is not None and m.GetNumAtoms() > 0:
            for step in range(10):
              rsm = Chem.MolToSmiles(m, rootedAtAtom=np.random.randint(
                  0, m.GetNumAtoms()), canonical=False)
              arr.append(rsm)
              if RETRAIN == "True" and canon != "":
                canon_pairs.append([rsm, canon])
          else:
            arr.append(mol)

            if RETRAIN == "True" and canon != "":
              canon_pairs.append([mol, canon])

      else:

        arr.append(mol)
        m = Chem.MolFromSmiles(mol)
        if m is not None:
          canon = Chem.MolToSmiles(m)

        if RETRAIN == "True" and canon != "":
          canon_pairs.append([mol, canon])

    except:
      arr.append(mol)

    vals = np.zeros(len(props), dtype=np.float32)
    mask = np.zeros(len(props), dtype=np.int8)

    for prop in props:
      idx = prop
      icsv = props[prop][0]
      s = row[icsv].strip()
      if s == '':
        val = 0
      else:
        val = float(s)
        mask[idx] = 1
      vals[idx] = val

    arr = list(set(arr))
    for step in range(len(arr)):
      DS.append([arr[step], np.copy(vals), mask])

  findBoundaries(DS)

  return DS

def gen_data(data):

  batch_size = len(data)

  # search for max lengths
  nl = len(data[0][0])
  for i in range(1, batch_size, 1):
    nl_a = len(data[i][0])
    if nl_a > nl:
      nl = nl_a

  nl = nl + CONV_OFFSET

  x = np.zeros((batch_size, nl), np.int8)
  mx = np.zeros((batch_size, nl), np.int8)

  z = []
  ym = []

  for i in range(len(props)):
    z.append(np.zeros((batch_size, 1), np.float32))
    ym.append(np.zeros((batch_size, 1), np.int8))

  for cnt in range(batch_size):

    n = len(data[cnt][0])
    for i in range(n):
      x[cnt, i] = char_to_ix[data[cnt][0][i]]
    mx[cnt, :i + 1] = 1

    for i in range(len(props)):
      z[i][cnt] = data[cnt][1][i]
      ym[i][cnt] = data[cnt][2][i]

  d = [x, mx]

  for i in range(len(props)):
    d.extend([ym[i]])

  return d, z

def data_generator(ds):

  data = []
  while True:
    for i in range(len(ds)):
      data.append(ds[i])
      if len(data) == BATCH_SIZE:
        try:
          yield gen_data(data)
          data = []
        except StopIteration:
          return
    if len(data) > 0:
      try:
        yield gen_data(data)
        data = []
      except StopIteration:
        return
    return

def buildNetwork():

  unfreeze = False

  l_in = layers.Input(shape=(None,))
  l_mask = layers.Input(shape=(None,))

  l_ymask = []
  for i in range(len(props)):
    l_ymask.append(layers.Input(shape=(1, )))

  # transformer part
  # positional encodings for product and reagents, respectively
  l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask)
  l_left_mask = MaskLayerLeft()(l_mask)

  # encoder
  l_voc = layers.Embedding(
      input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=None, trainable=unfreeze)
  l_embed = layers.Add()([l_voc(l_in), l_pos])

  for layer in range(n_block):

    # self attention
    l_o = [SelfLayer(EMBEDDING_SIZE, KEY_SIZE, trainable=unfreeze)(
        [l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)]

    l_con = layers.Concatenate()(l_o)
    l_dense = layers.TimeDistributed(layers.Dense(
        EMBEDDING_SIZE, trainable=unfreeze), trainable=unfreeze)(l_con)
    if unfreeze == True:
      l_dense = layers.Dropout(rate=0.1)(l_dense)
    l_add = layers.Add()([l_dense, l_embed])
    l_att = LayerNormalization(trainable=unfreeze)(l_add)

    # position-wise
    l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu',
                         trainable=unfreeze)(l_att)
    l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1, trainable=unfreeze)(l_c1)
    if unfreeze == True:
      l_c2 = layers.Dropout(rate=0.1)(l_c2)
    l_ff = layers.Add()([l_att, l_c2])
    l_embed = LayerNormalization(trainable=unfreeze)(l_ff)

  # end of Transformer's part
  l_encoder = l_embed

  # text-cnn part
  # https://github.com/deepchem/deepchem/blob/b7a6d3d759145d238eb8abaf76183e9dbd7b683c/deepchem/models/tensorgraph/models/text_cnn.py

  l_in2 = layers.Input(shape=(None, EMBEDDING_SIZE))

  kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
  num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

  l_pool = []
  for i in range(len(kernel_sizes)):
    l_conv = layers.Conv1D(num_filters[i], kernel_size=kernel_sizes[i], padding='valid',
                           kernel_initializer='normal', activation='relu')(l_in2)
    l_maxpool = layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(l_conv)
    l_pool.append(l_maxpool)

  l_cnn = layers.Concatenate(axis=1)(l_pool)
  l_cnn_drop = layers.Dropout(rate=0.25)(l_cnn)

  # dense part
  l_dense = layers.Dense(N_HIDDEN_CNN, activation='relu')(l_cnn_drop)

  # https://github.com/ParikhKadam/Highway-Layer-Keras
  transform_gate = layers.Dense(units=N_HIDDEN_CNN, activation="sigmoid",
                                bias_initializer=tf.keras.initializers.Constant(-1))(l_dense)

  carry_gate = layers.Lambda(
      lambda x: 1.0 - x, output_shape=(N_HIDDEN_CNN,))(transform_gate)
  transformed_data = layers.Dense(
      units=N_HIDDEN_CNN, activation="relu")(l_dense)
  transformed_gated = layers.Multiply()([transform_gate, transformed_data])
  identity_gated = layers.Multiply()([carry_gate, l_dense])

  l_highway = layers.Add()([transformed_gated, identity_gated])

  # Because of multitask we have here a few different outputs and a custom loss.

  def mse_loss(prop):
    def loss(y_true, y_pred):
      y2 = y_true * l_ymask[prop] + y_pred * (1 - l_ymask[prop])
      return tf.keras.losses.mse(y2, y_pred)
    return loss

  def binary_loss(prop):
    def loss(y_true, y_pred):
      y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
      r = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred)
      r = -tf.reduce_mean(r * l_ymask[prop])
      return r
    return loss

  l_out = []
  losses = []
  for prop in props:
    if props[prop][2] == "regression":
      l_out.append(layers.Dense(1, activation='linear',
                                name="Regression-" + props[prop][1])(l_highway))
      losses.append(mse_loss(prop))
    else:
      l_out.append(layers.Dense(1, activation='sigmoid',
                                name="Classification-" + props[prop][1])(l_highway))
      losses.append(binary_loss(prop))

  l_input = [l_in2]
  l_input.extend(l_ymask)

  mdl = tf.keras.Model(l_input, l_out)
  mdl.compile(optimizer='adam', loss=losses)

  # mdl.summary()

  K.set_value(mdl.optimizer.lr, 1.0e-4)

  # so far we do not train the encoder part of the model.
  encoder = tf.keras.Model([l_in, l_mask], l_encoder)
  encoder.compile(optimizer='adam', loss='mse')
  encoder.set_weights(np.load(f"{path}/embeddings.npy", allow_pickle=True))

  # encoder.summary()

  return mdl, encoder

# Transformer Model for canonization task

def Smi2Smi():

  # product
  l_in = layers.Input(shape=(None,))
  l_mask = layers.Input(shape=(None,))

  # reagents
  l_dec = layers.Input(shape=(None,))
  l_dmask = layers.Input(shape=(None,))

  # positional encodings for product and reagents, respectively
  l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask)
  l_dpos = PositionLayer(EMBEDDING_SIZE)(l_dmask)

  l_emask = MaskLayerRight()([l_dmask, l_mask])
  l_right_mask = MaskLayerTriangular()(l_dmask)
  l_left_mask = MaskLayerLeft()(l_mask)

  # encoder
  l_voc = layers.Embedding(
      input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=None)

  l_embed = layers.Add()([l_voc(l_in), l_pos])
  l_embed = layers.Dropout(rate=0.1)(l_embed)

  for layer in range(n_block):

    # self attention
    l_o = [SelfLayer(EMBEDDING_SIZE, KEY_SIZE)(
        [l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)]

    l_con = layers.Concatenate()(l_o)
    l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(l_con)
    l_drop = layers.Dropout(rate=0.1)(l_dense)
    l_add = layers.Add()([l_drop, l_embed])
    l_att = LayerNormalization()(l_add)

    # position-wise
    l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu')(l_att)
    l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1)(l_c1)
    l_drop = layers.Dropout(rate=0.1)(l_c2)
    l_ff = layers.Add()([l_att, l_drop])
    l_embed = LayerNormalization()(l_ff)

  # bottleneck
  l_encoder = l_embed

  l_embed = layers.Add()([l_voc(l_dec), l_dpos])
  l_embed = layers.Dropout(rate=0.1)(l_embed)

  for layer in range(n_block):

    # self attention
    l_o = [SelfLayer(EMBEDDING_SIZE, KEY_SIZE)(
        [l_embed, l_embed, l_embed, l_right_mask]) for i in range(n_self)]

    l_con = layers.Concatenate()(l_o)
    l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(l_con)
    l_drop = layers.Dropout(rate=0.1)(l_dense)
    l_add = layers.Add()([l_drop, l_embed])
    l_att = LayerNormalization()(l_add)

    # attention to the encoder
    l_o = [SelfLayer(EMBEDDING_SIZE, KEY_SIZE)(
        [l_att, l_encoder, l_encoder, l_emask]) for i in range(n_self)]
    l_con = layers.Concatenate()(l_o)
    l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(l_con)
    l_drop = layers.Dropout(rate=0.1)(l_dense)
    l_add = layers.Add()([l_drop, l_att])
    l_att = LayerNormalization()(l_add)

    # position-wise
    l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu')(l_att)
    l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1)(l_c1)
    l_drop = layers.Dropout(rate=0.1)(l_c2)
    l_ff = layers.Add()([l_att, l_drop])
    l_embed = LayerNormalization()(l_ff)

  l_out = layers.TimeDistributed(layers.Dense(vocab_size,
                                              use_bias=False))(l_embed)

  mdl = tf.keras.Model([l_in, l_mask, l_dec, l_dmask], l_out)

  def masked_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y_true, logits=y_pred)
    mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32')
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    loss = K.mean(loss)
    return loss

  def masked_acc(y_true, y_pred):
    mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32')
    eq = K.cast(K.equal(K.argmax(y_true, axis=-1),
                        K.argmax(y_pred, axis=-1)), 'float32')
    eq = tf.reduce_sum(eq * mask, -1) / tf.reduce_sum(mask, -1)
    eq = K.mean(eq)
    return eq

  mdl.compile(optimizer='adam', loss=masked_loss,
              metrics=['accuracy', masked_acc])

  mdl_enc = tf.keras.Model([l_in, l_mask], l_encoder)
  mdl_enc.compile(optimizer="adam", loss="categorical_crossentropy")

  # mdl.summary()

  return mdl, mdl_enc

def smi_gen_data(data):

  batch_size = len(data)

  # search for max lengths
  left = []
  right = []
  for line in data:
    left.append(line[0].strip())
    right.append(line[1].strip())

  nl = len(left[0])
  nr = len(right[0])
  for i in range(1, batch_size, 1):
    nl_a = len(left[i])
    nr_a = len(right[i])
    if nl_a > nl:
      nl = nl_a
    if nr_a > nr:
      nr = nr_a

  # add start and end symbols
  nl += 2
  nr += 1

  # products
  x = np.zeros((batch_size, nl), np.int8)
  mx = np.zeros((batch_size, nl), np.int8)

  # reactants
  y = np.zeros((batch_size, nr), np.int8)
  my = np.zeros((batch_size, nr), np.int8)

  # for output
  z = np.zeros((batch_size, nr, vocab_size), np.int8)

  for cnt in range(batch_size):
    product = "^" + left[cnt] + "$"
    reactants = "^" + right[cnt]

    reactants += "$"
    for i, p in enumerate(product):
      x[cnt, i] = char_to_ix[p]

    mx[cnt, :i + 1] = 1
    for i in range((len(reactants) - 1)):
      y[cnt, i] = char_to_ix[reactants[i]]
      z[cnt, i, char_to_ix[reactants[i + 1]]] = 1

    my[cnt, :i + 1] = 1

  return [x, mx, y, my], z


if __name__ == "__main__":

  device_str = "GPU" + str(DEVICE)

  if TRAIN == "True":

    DS = analyzeDescrFile(TRAIN_FILE)

    if len(canon_pairs) > 0:
      random.shuffle(canon_pairs)

      smi2smi, smi_encoder = Smi2Smi()

      if CHIRALITY == "True":
        smi2smi.load_weights("pretrained/canonization.h5")
      else:
        smi2smi.load_weights("pretrained/canonization-nochiral.h5")

      epochs_to_save = [6, 7, 8, 9]
      smi_batch = 32

      class GenCallback(tf.keras.callbacks.Callback):
        def __init__(self, eps=1e-6, **kwargs):
          self.steps = 0
          self.warm = 16000
          self.steps = self.warm + 30

        def on_batch_begin(self, batch, logs={}):
          self.steps += 1
          lr = 20.0 * min(1.0, self.steps / self.warm) / \
              max(self.steps, self.warm)
          K.set_value(self.model.optimizer.lr, lr)

        def on_epoch_end(self, epoch, logs={}):
          if epoch in epochs_to_save:
            smi2smi.save_weights(
                f"{path}/tr-" + str(epoch) + ".h5", save_format="h5")

      def smi2smi_generator():
        lines = []
        while True:
          for i in range(len(canon_pairs)):
            if len(lines) == smi_batch:
              yield smi_gen_data(lines)
              lines = []
            lines.append(canon_pairs[i])

      callback = [GenCallback()]
      history = smi2smi.fit_generator(generator=smi2smi_generator(),
                                      steps_per_epoch=int(
          math.ceil(len(canon_pairs) / smi_batch)),
          epochs=10,
          use_multiprocessing=False,
          shuffle=True,
          callbacks=callback)

      f = []

      for i in epochs_to_save:
        f.append(h5py.File(f"{path}/tr-" + str(i) + ".h5", "r+"))

      keys = list(f[0].keys())
      for key in keys:
        groups = list(f[0][key])
        if len(groups):
          for group in groups:
            items = list(f[0][key][group].keys())
            for item in items:
              data = []
              for i in range(len(f)):
                data.append(f[i][key][group][item])
              avg = np.mean(data, axis=0)
              del f[0][key][group][item]
              f[0][key][group].create_dataset(item, data=avg)
      for fp in f:
        fp.close()

      for i in epochs_to_save[1:]:
        os.remove(f"{path}/tr-" + str(i) + ".h5")
      os.rename(f"{path}/tr-" +
                str(epochs_to_save[0]) + ".h5", f"{path}/final.h5")

      # extract embeddings
      smi2smi.load_weights(f"{path}/final.h5")
      w = smi_encoder.get_weights()
      np.save(f"{path}/embeddings.npy", w)
      os.remove(f"{path}/final.h5")

    else:
      if CHIRALITY == "True":
        shutil.copy("pretrained/embeddings.npy", f"{path}/embeddings.npy")
      else:
        shutil.copy("pretrained/embeddings-nochiral.npy",
                    f"{path}/embeddings.npy")

    # end of pretraining

    mdl, encoder = buildNetwork()

    nall = len(DS)

    inds = np.arange(nall)

    def data_generator2(dsc):
      while True:
        for i in range(len(dsc)):
          yield dsc[i][0], dsc[i][1]

    if EARLY_STOPPING == 0:
      np.random.shuffle(inds)

      d = [DS[x] for x in inds]
      all_generator = data_generator(d)

      DSC_ALL = []
      for x, y in all_generator:
        z = encoder.predict(x)
        DSC_ALL.append((z, y))

      all_generator = data_generator2(DSC_ALL)

    else:
      np.random.shuffle(inds)
      ntrain = int(EARLY_STOPPING * nall)

      inds_train = inds[:ntrain]
      inds_valid = inds[ntrain:]

      DS_train = [DS[x] for x in inds_train]
      DS_valid = [DS[x] for x in inds_valid]

      train_generator = data_generator(DS_train)
      valid_generator = data_generator(DS_valid)

      DSC_TRAIN = []
      DSC_VALID = []

      # calculate "descriptors"
      for x, y in train_generator:

        z = encoder.predict([x[0], x[1]])
        d = [z]
        for i in range(len(props)):
          d.extend([x[i + 2]])
        DSC_TRAIN.append((d, y))

      for x, y in valid_generator:
        z = encoder.predict([x[0], x[1]])
        d = [z]
        for i in range(len(props)):
          d.extend([x[i + 2]])
        DSC_VALID.append((d, y))

      train_generator = data_generator2(DSC_TRAIN)
      valid_generator = data_generator2(DSC_VALID)

    class MessagerCallback(tf.keras.callbacks.Callback):

      def __init__(self, **kwargs):
        self.steps = 0
        self.warm = 64

        self.early_max = 0.2 * NUM_EPOCHS if EARLY_STOPPING > 0 else NUM_EPOCHS
        self.early_best = 0.0
        self.early_count = 0

        if EARLY_STOPPING > 0:
          self.valid_gen = data_generator2(DSC_VALID)
          self.train_gen = data_generator2(DSC_TRAIN)
        else:
          self.train_gen = data_generator2(DSC_ALL)

      def on_batch_begin(self, batch, logs={}):
        self.steps += 1
        if FIXED_LEARNING_RATE == "False":
          lr = 1.0 * min(1.0, self.steps / self.warm) / \
              max(self.steps, self.warm)
          if lr < 1e-4:
            lr = 1e-4
          K.set_value(self.model.optimizer.lr, lr)

      def on_epoch_end(self, epoch, logs={}):
        if EARLY_STOPPING > 0:
          print("MESSAGE: train score: {} / validation score: {} / at epoch: {} {} ".format(round(float(logs["loss"]), 7),
                                                                                            round(float(logs["val_loss"]), 7), epoch + 1, device_str))
          early = float(logs["val_loss"])
          if(epoch == 0):
            self.early_best = early
          else:
            if early < self.early_best:
              self.early_count = 0
              self.early_best = early
            else:
              self.early_count += 1
              if self.early_count > self.early_max:
                self.model.stop_training = True
              return

        else:
          print("MESSAGE: train score: {} / at epoch: {} {} ".format(round(float(logs["loss"]), 5),
                                                                     epoch + 1, device_str))

        if EARLY_STOPPING == 0 and epoch >= (NUM_EPOCHS - AVERAGING - 1):
          self.model.save_weights(f"{path}/e-" + str(epoch) + ".h5")

        if os.path.exists("stop"):
          self.model.stop_training = True
        return

    if EARLY_STOPPING > 0:
      history = mdl.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(DSC_TRAIN),
                                  epochs=NUM_EPOCHS,
                                  validation_data=valid_generator,
                                  validation_steps=len(DSC_VALID),
                                  use_multiprocessing=False,
                                  shuffle=True,
                                  verbose=0,
                                  callbacks=[ModelCheckpoint("model/", monitor='val_loss',
                                                             save_best_only=True, save_weights_only=True,
                                                             mode='auto', period=1),
                                             MessagerCallback()])

      mdl.load_weights(f"{path}/model/")  # restoring best saved model
      mdl.save_weights(f"{path}/model.h5")

    else:
      history = mdl.fit_generator(generator=all_generator,
                                  steps_per_epoch=len(DSC_ALL),
                                  epochs=NUM_EPOCHS,
                                  use_multiprocessing=False,
                                  shuffle=True,
                                  verbose=0,
                                  callbacks=[MessagerCallback()])

      f = []

      for i in range(NUM_EPOCHS - AVERAGING - 1, NUM_EPOCHS):
        f.append(h5py.File(f"{path}/e-" + str(i) + ".h5", "r+"))

      keys = list(f[0].keys())
      for key in keys:
        groups = list(f[0][key])
        if len(groups):
          for group in groups:
            items = list(f[0][key][group].keys())
            for item in items:
              data = []
              for i in range(len(f)):
                data.append(f[i][key][group][item])
              avg = np.mean(data, axis=0)
              del f[0][key][group][item]
              f[0][key][group].create_dataset(item, data=avg)
      for fp in f:
        fp.close()

      for i in range(NUM_EPOCHS - AVERAGING, NUM_EPOCHS):
        os.remove(f"{path}/e-" + str(i) + ".h5")
      os.rename(f"{path}/e-" + str(NUM_EPOCHS - AVERAGING - 1) +
                ".h5", f"{path}/model.h5")

    with open(f'{path}/model.pkl', 'wb') as f:
      pickle.dump(props, f)
    tar = tarfile.open(MODEL_FILE, "w:gz")
    tar.add(f"{path}/model.pkl")
    tar.add(f"{path}/model.h5")
    tar.add(f"{path}/embeddings.npy")
    tar.close()

    if EARLY_STOPPING > 0:
      shutil.rmtree("model/")

    os.remove(f"{path}/model.pkl")
    os.remove(f"{path}/model.h5")
    os.remove(f"{path}/embeddings.npy")

  elif TRAIN == "False":

    tar = tarfile.open(MODEL_FILE)
    tar.extractall(path=path)
    tar.close()

    props = pickle.load(open(f"{path}/model.pkl", "rb"))

    mdl, encoder = buildNetwork()
    mdl.load_weights(f"{path}/model.h5")

    first_row = FIRST_LINE
    DS = []

    fp = open(RESULT_FILE, "w")
    for prop in props:
      print(props[prop][1], end=",", file=fp)
    print("", file=fp)

    ind_mol = 0

    if CANONIZE == 'True':
      remover = SaltRemover.SaltRemover()
      for row in csv.reader(open(APPLY_FILE, "r")):
        if first_row:
          first_row = False
          continue

        mol = row[ind_mol]
        g_mol = set(mol)
        g_left = g_mol - g_chars

        arr = []
        if len(g_left) == 0:
          try:
            with suppress_stderr():
              m = Chem.MolFromSmiles(mol)
              m = remover.StripMol(m)
              if m is not None and m.GetNumAtoms() > 0:
                for step in range(10):
                  arr.append(Chem.MolToSmiles(m, rootedAtAtom=np.random.randint(
                      0, m.GetNumAtoms()), canonical=False))
              else:
                arr.append(mol)
          except:
            arr.append(mol)
        else:
          for i in range(len(props)):
            print("error", end=",", file=fp)

        z = np.zeros(len(props), dtype=np.float32)
        ymask = np.ones(len(props), dtype=np.int8)

        d = []
        for i in range(len(arr)):
          d.append([arr[i], z, ymask])

        x, y = gen_data(d)
        internal = encoder.predict([x[0], x[1]])

        p = [internal]
        for i in range(len(props)):
          p.extend([x[i + 2]])

        y = mdl.predict(p)
        res = np.zeros(len(props))

        for prop in props:
          if len(props) == 1:
            res[prop] = np.mean(y)
          else:
            res[prop] = np.mean(y[prop])
          if props[prop][2] == "regression":
            res[prop] = (res[prop] - 0.9) / 0.8 * (props[prop]
                                                   [4] - props[prop][3]) + props[prop][4]
          print(res[prop], end=",", file=fp)
        print("", file=fp)

    else:

      arr = []
      e = []

      for row in csv.reader(open(APPLY_FILE, "r")):
        if first_row:
          first_row = False
          continue

        mol = row[ind_mol]
        g_mol = set(mol)
        g_left = g_mol - g_chars

        if len(g_left) == 0:
          e.append(1)
          arr.append(mol)
        else:
          e.append(0)
          arr.append("CC")

        if (len(arr) == BATCH_SIZE):

          z = np.zeros(len(props), dtype=np.float32)
          ymask = np.ones(len(props), dtype=np.int8)

          d = []
          for i in range(len(arr)):
            d.append([arr[i], z, ymask])

          x, y = gen_data(d)
          internal = encoder.predict([x[0], x[1]])

          p = [internal]
          for i in range(len(props)):
            p.extend([x[i + 2]])

          y = mdl.predict(p)
          res = np.zeros(len(props))

          for i in range(len(arr)):
            if e[i] == 0:
              for prop in props:
                print("error", end=",", file=fp)
              print("", file=fp)
              continue

            for prop in props:
              if len(props) == 1:
                res = y[i][prop]
              else:
                res = y[prop][i][0]

              if props[prop][2] == "regression":
                res = (res - 0.9) / 0.8 * \
                    (props[prop][4] - props[prop][3]) + props[prop][4]
              print(res, end=",", file=fp)
            print("", file=fp)

          arr = []
          e = []

      if len(arr):

        z = np.zeros(len(props), dtype=np.float32)
        ymask = np.ones(len(props), dtype=np.int8)

        d = []
        for i in range(len(arr)):
          d.append([arr[i], z, ymask])

        x, y = gen_data(d)
        internal = encoder.predict([x[0], x[1]])

        p = [internal]
        for i in range(len(props)):
          p.extend([x[i + 2]])

        y = mdl.predict(p)
        res = np.zeros(len(props))

        for i in range(len(arr)):
          if e[i] == 0:
            for prop in props:
              print("error", end=",", file=fp)
            print("", file=fp)
            continue

          for prop in props:
            if len(props) == 1:
              res = y[i][prop]
            else:
              res = y[prop][i][0]

            if props[prop][2] == "regression":
              res = (res - 0.9) / 0.8 * \
                  (props[prop][4] - props[prop][3]) + props[prop][4]
            print(res, end=",", file=fp)
          print("", file=fp)

    fp.close()

    os.remove(f"{path}/model.pkl")
    os.remove(f"{path}/model.h5")
    os.remove(f"{path}/embeddings.npy")
    import pandas as pd
    df = pd.read_csv(RESULT_FILE)
    df = pd.DataFrame(df["property"])
    df.to_csv(f"{path}/results.csv", index=False)
  print("Relax!")
