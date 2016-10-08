#! /usr/bin/env python

import sys
import os
import time
import numpy as np
import scipy
from utils import *
from datetime import datetime
from gru_theano import GRUTheano

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.000625"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "2000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "256"))
NEPOCH = int(os.environ.get("NEPOCH", "200"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE", "Image-Recognition-L" + str(LEARNING_RATE) + "-10images-256.dat.npz")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "200"))

model = load_model_embeddings_theano('./data/pretrained.npz')
gru = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
#gru = load_model_parameters_theano('Image-Recognition-L0.000625-10images-256.dat.npz')
if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

images = []
indices = []
processedImages = []

gru.E = model.E

# Load data
#x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
word_to_index = load_vocab()

with open("image-labels-10", "r") as f:
  for line in f:
    lineParts = line.rstrip('\n').rstrip('\r').split(' ')
    images.append(scipy.misc.imread("Training-Images/" + lineParts[0]))
    indices.append(word_to_index[lineParts[1]])

processedImage = np.zeros(shape=(32,128))
processedImage2 = np.zeros(shape=(32, 128))

for idx, image in enumerate(images):
  processedImages.append(np.zeros(shape=(32,128)))
  for i in xrange(32):
    processedImages[idx][i] = image[i].flatten()
    for j in xrange(128):
      processedImages[idx][i][j] = processedImages[idx][i][j] / 255.0

# Build model

# Print SGD step time
t1 = time.time()
gru.sgd_step(indices[0], processedImages[0], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
print "Loss:"
print gru.calculate_loss(indices[0], processedImages[0])
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen, s, whichepoch, whichitem):
  print("CALLBACK ACTIVATED")
  dt = datetime.now().isoformat()
  loss = gru.calculate_loss(indices[whichitem], processedImages[whichitem])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  print(s)
  save_model_parameters_theano(gru, MODEL_OUTPUT_FILE)
  outputImage = np.asarray(s)[0]
  print("input image shape:")
  print(image.shape)
  print("output image shape:")
  print(outputImage.shape)
  recalledImage = np.zeros(shape=(32, 32, 4))
  for i in xrange(32):
    for j in xrange(128):
      outputImage[i][j] = outputImage[i][j] * 255.0
    recalledImage[i] = outputImage[i].reshape(32, 4)
  scipy.misc.imsave("outputs" + str(LEARNING_RATE) + "-10images-256/output-" + str(whichepoch) + ".png", recalledImage)
  print("\n")
  sys.stdout.flush()

for epoch in range(NEPOCH):
  train_with_sgd(gru, indices, processedImages, LEARNING_RATE, nepoch=200, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback, whichepoch=epoch)

outputImages = []

for i in xrange(len(images)):
  outputImages.append(np.asarray(gru.sgd_step(indices[i], processedImages[i], LEARNING_RATE))[0])

recalledImages = []

for idx, image in enumerate(outputImages):
  recalledImages.append(np.zeros(shape=(32, 32, 4)))
  for i in xrange(32):
    for j in xrange(128):
      outputImages[idx][i][j] = outputImages[idx][i][j] * 255.0
    recalledImages[idx][i] = outputImages[idx][i].reshape(32, 4)
  scipy.misc.imsave("outputs" + str(LEARNING_RATE) + "-10images-256/recalled-" + str(idx) + ".png", recalledImages[idx])

