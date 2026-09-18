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
"""Input examples, features, and dataset processors for the classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import tensorflow as tf

from bert import tokenization
from bert.utils import *


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
