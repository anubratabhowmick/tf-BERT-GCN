# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import modeling
import optimization
import tokenization
import tensorflow as tf
from utils import *

flags = tf.flags
FLAGS = flags.FLAGS

''' USE '''

flags.DEFINE_string(
    "model", None,
    "Model Name"
    "for the task.")

flags.DEFINE_string(
    "dataset", None,
    "Dataset"
    "for the task.")

flags.DEFINE_integer(
    "frequency", None,
    "Dataset frequency"
    "for the task.")

flags.DEFINE_string(
    "gpu", '0',
    "Use Gpu"
    "for the task.")

flags.DEFINE_integer(
    "year", 2013,
    "split Dataset train/test"
    "for the task.")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The metric_result directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

''' NOT USE '''

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, meta1=None, meta2=None): #Added by Anubrata
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta1 = meta1 #Added by Anubrata
        self.meta2 = meta2 #Added by Anubrata


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire metric_result data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True,
                 meta1=None,
                 meta2=None):
                 #Added by Anubrata
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.meta1 = meta1 # Added by Anubrata
        self.meta2 = meta2 # Added by Anubrata


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class BaseBertProcessor(DataProcessor):

    def __init__(self, dataset, frequency, seq_len, year):

        self.column = ['left_citated_text', 'right_citated_text', 'target_id', 'source_id', 'target_year',
                       'target_author']
        self.frequency = frequency
        self.seq_len = seq_len
        self.bert_column = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String']
        self.year = year
        self.dataset = dataset
        self.flag = 'bert_base' # Added by Anubrata to bypass column adding problem
        self.train_df, self.test_df, self.lb = load_data(self.dataset, self.column, self.frequency, self.seq_len,
                                                         self.year, self.bert_column, self.flag)
        self.meta1_shape = 1 #Added by Anubrata
        self.meta2_shape = 1 #Added by Anubrata 

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.train_df, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.test_df, "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(len(self.lb.classes_))]

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for data in df.values:
            guid = "%s-%s" % (set_type, data[1])
            text_a = tokenization.convert_to_unicode(data[3])
            text_b = tokenization.convert_to_unicode(data[4])
            label = data[0]
            meta1 = [0] * self.meta1_shape #Added by Anubrata
            meta2 = [0] * self.meta2_shape #Added by Anubrata
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta1=meta1, meta2=meta2)) #Added by Anubrata
        return examples


class BertGcnProcessor(DataProcessor):
    def __init__(self, dataset, frequency, seq_len, year):

        # if data_name == 'AAN':
        #     meta_data_name = 'AAN_{}_gcn_100d.pkl'
        # else:
        #     meta_data_name = 'PeerRead_{}_gcn_100d.pkl'
        self.meta_dataset_name = '{}_vgae_paperfeatureless_768d_encoded.pkl'.format(dataset)
        self.column = ['left_citated_text', 'right_citated_text', 'target_id', 'source_id', 'target_year',
                       'target_author', 'source_author'] # Added by Anubrata
        self.frequency = frequency
        self.bert_column = ['Quality', '#1 ID', '#2 ID', '#1 String', '#2 String', 'target_id']
        self.year = year
        self.dataset = dataset
        self.seq_len = seq_len
        self.flag = 'bert_gcn' # Added by Anubrata to bypass column adding problem
        self.train_df, self.test_df, self.lb = load_data(self.dataset, self.column, self.frequency, self.seq_len,
                                                         self.year, self.bert_column, self.flag)
        #self.gcn_data = load_pickle(FLAGS.data_dir, self.meta_dataset_name)

        # Added by Anubrata Start
        self.author_train, self.author_test, self.embedding_author, self.node2id_author = get_gcn_author_data(self.train_df, self.test_df,
                                                                                   './pre_train/gcn',
                                                                                   '{}_gcn_pretrain_author.pkl'.format(dataset))

        self.train_df = self.train_df[self.bert_column]
        self.test_df = self.test_df[self.bert_column]
        # Added by Anubrata End


        self.gcn_train, self.gcn_test, self.embedding, self.node2id = get_gcn_data(self.train_df, self.test_df,
                                                                                   './pre_train/gcn',
                                                                                    # self.meta_dataset_name)
                                                                                   '{}_gcn_pretrain.pkl'.format(dataset))
        self.meta1_shape = 1 #Added by Anubrata
        self.meta2_shape = 1 #Added by Anubrata

    def get_train_examples(self, data_dir):
        """See base class."""
        #return self._create_examples(self.train_df, self.gcn_data, "train")
        return self._create_examples(self.train_df, self.gcn_train, self.author_train, "train") # Added by Anubrata

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(self.test_df, self.gcn_data, "test")
        return self._create_examples(self.test_df, self.gcn_test, self.author_test,  "test") # Added by Anubrata

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(len(self.lb.classes_))]

    def _create_examples(self, df, gcn_data_value, author_data_values, set_type): # Added by Anubrata
        """Creates examples for the training and dev sets."""
        examples = []
        # gcn_examples = gcn_data_value[df.index]

        for data, gcn_example, author_data_values in zip(df.values, gcn_data_value, author_data_values): # Added by Anubrata
            guid = "%s-%s" % (set_type, data[1])
            text_a = tokenization.convert_to_unicode(data[3])
            text_b = tokenization.convert_to_unicode(data[4])
            label = data[0]
            meta1 = gcn_example # Added by Anubrata gcn_example
            meta2 =  author_data_values # Added by Anubrata
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta1=meta1, meta2=meta2)) # Added by Anubrata
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False,
        )

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[str(example.label)]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("meta1: %s" % (example.meta1)) # Added by Anubrata
        tf.logging.info("meta2: %s" % (example.meta2)) # Added by Anubrata

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True,
        meta1=example.meta1, # Added by Anubrata
        meta2=example.meta2) # Added by Anubrata
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        # features["meta"] = create_float_feature(feature.meta)
        # print('meta1: ', feature.meta1)
        if 0 in feature.meta1 and 0 in feature.meta2:
            features["meta1"] = create_int_feature(feature.meta1) 
            features["meta2"] = create_int_feature(feature.meta2)
        else:
            features["meta1"] = create_int_feature(feature.meta1.astype(int)) # Added by Anubrata
            features["meta2"] = create_int_feature(feature.meta2.astype(int))# Added by Anubrata
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, meta1_length, meta2_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        # "meta": tf.FixedLenFeature([meta_length], tf.float32),
        "meta1": tf.FixedLenFeature([meta1_length], tf.int64), # Added by Anubrata
        "meta2": tf.FixedLenFeature([meta2_length], tf.int64), # Added by Anubrata
        #"meta": tf.FixedLenSequenceFeature([meta_length],  tf.int64, allow_missing=True, default_value=0) # Added by Anubrata
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:

        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, metas1=None, metas2=None, model='bert_base', dataset='AAN'):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        metas1=metas1, # Added by Anubrata
        metas2=metas2, # Added by Anubrata
        model=model,
        dataset=dataset
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level metric_result, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")

        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        metas1 = features["meta1"] # Added by Anubrata
        metas2 = features["meta2"] # Added by Anubrata
        print('Anubrata run metas 1: ', metas1) # Added by Anubrata
        print('Anubrata run metas 2: ', metas2) # Added by Anubrata
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, metas1, metas2, FLAGS.model, FLAGS.dataset)

        print('Create model done.....')
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        # elif mode == tf.estimator.ModeKeys.EVAL:
        #
        #   def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        #     predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        #     accuracy = tf.metrics.accuracy(
        #         labels=label_ids, predictions=predictions, weights=is_real_example)
        #     loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        #     return {
        #         "eval_accuracy": accuracy,
        #         "eval_loss": loss,
        #     }
        #
        #   eval_metrics = (metric_fn,
        #                   [per_example_loss, label_ids, logits, is_real_example])
        #   output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        #       mode=mode,
        #       loss=total_loss,
        #       eval_metrics=eval_metrics,
        #       scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    all_label_metas1 = [] # Added by Anubrata
    all_label_metas2 = [] # Added by Anubrata
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)
        all_label_metas1.append(feature.meta1) # Added by Anubrata
        all_label_metas2.append(feature.meta2) # Added by Anubrata

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
            # "meta":
            #     tf.constant(all_label_metas, shape=[num_examples], dtype=tf.float32)
            "meta1":
                tf.constant(all_label_metas1, shape=[num_examples], dtype=tf.int32), # Added by Anubrata
            "meta2":
                tf.constant(all_label_metas2, shape=[num_examples], dtype=tf.int32) # Added by Anubrata
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    processors = {
        "bert_base": BaseBertProcessor,
        "bert_gcn": BertGcnProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    model_name = FLAGS.model
    data_dir = FLAGS.data_dir
    frequency = FLAGS.frequency
    seq_len = FLAGS.max_seq_length
    dataset = FLAGS.dataset
    year = FLAGS.year
    output_dir = FLAGS.output_dir
    experience_dir = os.path.join(output_dir, dataset, model_name, "f_{}_u_{}".format(frequency, seq_len))
    tf_output_dir = os.path.join(experience_dir, "tf_outputs")
    df_dir = os.path.join(experience_dir, "df")
    prediction_dir = os.path.join(experience_dir, "predictions")
    metric_dir = os.path.join(experience_dir, "metric")

    if tf.gfile.Exists(output_dir) == False:
        tf.gfile.MakeDirs(output_dir)
    if tf.gfile.Exists(experience_dir) == False:
        tf.gfile.MakeDirs(experience_dir)
    if tf.gfile.Exists(tf_output_dir) == False:
        tf.gfile.MakeDirs(tf_output_dir)
    if tf.gfile.Exists(df_dir) == False:
        tf.gfile.MakeDirs(df_dir)
    if tf.gfile.Exists(prediction_dir) == False:
        tf.gfile.MakeDirs(prediction_dir)
    if tf.gfile.Exists(metric_dir) == False:
        tf.gfile.MakeDirs(metric_dir)

    processor = processors[model_name](dataset, frequency, seq_len, year)
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=tf_output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    if FLAGS.do_train:
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        print('Entering for training...')
        train_file = os.path.join(tf_output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, processor.seq_len, tokenizer, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=processor.seq_len,
            is_training=True,
            drop_remainder=True,
            meta1_length=processor.meta1_shape, # Added by Anubrata
            meta2_length=processor.meta2_shape) # Added by Anubrata
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=tf_output_dir,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,  # 이 부분 의심
            config=run_config,  # 이부분도 의심
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(tf_output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                processor.seq_len, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=processor.seq_len,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            meta1_length=processor.meta1_shape, # Added by Anubrata
            meta2_length=processor.meta2_shape) # Added by Anubrata

        result = estimator.predict(input_fn=predict_input_fn)
        get_predictions(result, prediction_dir, 'test', num_actual_predict_examples)
        predictions = read_predictions(prediction_dir, 'test')
        multi_label_info = get_multi_label_info(processor.test_df.reset_index())
        write_pickle(processor.test_df, df_dir, 'test_df')
        y_true, label_predictions, dummy = convert_class_to_label(multi_label_info, predictions)
        write_pickle(y_true, prediction_dir, 'y_true')
        threshold = 0.0000000001

        if tf.gfile.Exists(metric_dir) == False:
            tf.gfile.MakeDirs(metric_dir)
        # write_spec(metric_dir, f, u, processor.seq_len)
        TOP_K = [5, 10, 30, 50, 80]

        for k in TOP_K:
            precision_value, recall_value = precision_recall_at_k(dummy, y_true, label_predictions, k, threshold)
            write_report(metric_dir, recall_value, k, method='recall', frequency=frequency, seq_len=seq_len)
        map_value = map_evaluate(dummy, y_true, label_predictions)
        mrr_value = mean_reciprocal_rank(y_true, label_predictions)
        write_report(metric_dir, mrr_value, top_k=None, method='mrr', frequency=frequency, seq_len=seq_len)
        write_report(metric_dir, map_value, top_k=None, method='map', frequency=frequency, seq_len=seq_len)




if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("frequency")
    tf.app.run()