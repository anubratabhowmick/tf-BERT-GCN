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
"""Training/prediction orchestration for the BERT(-GCN) classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from bert import modeling
from bert import tokenization
from bert.utils import *
from classifier.flags import FLAGS
from classifier.data import BaseBertProcessor, BertGcnProcessor
from classifier.model import model_fn_builder
from classifier.features import file_based_convert_examples_to_features, file_based_input_fn_builder


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
