# -*- coding: utf-8 -*-
# @Time    : 2020/4/4 01:00
# @Author  : Bruce
# @Email   : daishaobing@outlook.com
# @File    : text_classification_5
# @Project: BERT


import os

import tensorflow as tf
import tensorflow_datasets

# set memory auto growth to monitor the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    TFBertForSequenceClassification,
)

# load the self-defined dataloader
from glue_new import glue_convert_examples_to_features, glue_processors

# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 10
MAX_LEN = 40

TASK = "user"
data_dir = '.\\data'
bert_config_dir = '.\\bert\\bert-base-multilingual-cased-config.json'
bert_vocab_dir = '.\\bert\\bert-base-multilingual-cased-vocab.txt'
bert_dir = '.\\bert\\bert-base-multilingual-cased-tf_model.h5'



processor = glue_processors[TASK]()
num_labels = len(processor.get_labels())
print(num_labels)

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
config = BertConfig.from_pretrained(bert_config_dir, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(bert_vocab_dir)
model = TFBertForSequenceClassification.from_pretrained(bert_dir, config=config)


# make dataset from file
train_examples = processor.get_train_examples(data_dir=data_dir)
valid_examples = processor.get_dev_examples(data_dir=data_dir)


# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(train_examples, tokenizer, MAX_LEN, TASK)
valid_dataset = glue_convert_examples_to_features(valid_examples, tokenizer, MAX_LEN, TASK)

train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)



# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
if USE_AMP:
    # loss scaling is currently required when using mixed precision
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")


if num_labels == 1:
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=opt, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
train_steps = len(train_examples) // BATCH_SIZE
valid_steps = len(valid_examples) // EVAL_BATCH_SIZE

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=valid_dataset,
    validation_steps=valid_steps,
)

model.evaluate(valid_dataset)

# print the val_accuracy
import matplotlib.pyplot as plt
print(history.history['val_accuracy'])
t = range(len(history.history['accuracy']))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(t, acc, label='train_acc')
plt.plot(t, val_acc, label='val_acc')
plt.legend()
plt.grid()
plt.show()

# Save TF2 model
os.makedirs("./save_new/", exist_ok=True)
model.save_pretrained("./save_new/")


# 测试效果
if TASK == "user":
    # Load the TensorFlow model in PyTorch for inspection
    # This is to demo the interoperability between the two frameworks, you don't have to
    # do this in real life (you can run the inference on the TF model).
    pytorch_model = BertForSequenceClassification.from_pretrained("./save_new/", from_tf=True)

    # Quickly test a few predictions - Sentiment classifier
    sentence_1 = "送餐快,态度也特别好,辛苦啦谢谢"
    sentence_2 = "鸡肉死丁丁的，不好吃"
    inputs_1 = tokenizer.encode_plus(sentence_1, '', add_special_tokens=True, return_tensors="pt")
    inputs_2 = tokenizer.encode_plus(sentence_2, '', add_special_tokens=True, return_tensors="pt")

    pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
    pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
    print("sentiment of sentence_1 is", "positive" if pred_1 == 1 else "negative")
    print("sentiment of sentence_2 is", "positive" if pred_2 == 1 else "negative")

