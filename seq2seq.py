#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:48:16 2019

@author: clytie
"""

import tensorflow as tf
import numpy as np
import os
import math
import pickle

from tqdm import tqdm
from tensorboardX import SummaryWriter


def load_data(PATH=''):
    try:
        with open(PATH + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


def generator(inputs, targets, inputs_length, targets_length, batch_size):
    n_sample = len(inputs)
    index = np.arange(n_sample)
    np.random.shuffle(index)
    batch_datas = []
    for i in range(math.ceil(n_sample / batch_size)):
        span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
        span_index = index[span_index]
        batch_inputs_length = inputs_length[span_index]
        batch_inputs = inputs[span_index, : batch_inputs_length.max()]
        
        batch_targets_length = targets_length[span_index]
        batch_targets = targets[span_index, : batch_targets_length.max()]
        batch_datas.append((batch_inputs, batch_targets, batch_inputs_length, batch_targets_length))
    return batch_datas


class Seq2SeqModel(object):
    def __init__(self, vocab_size, hidden_size,
                 batch_size=64,
                 beam_search=3,
                 keep_prob=0.5,
                 max_grad_norm=1.0,
                 embedding_size=100,
                 lr_schedule=lambda x: max(0.05, (1 - x)) * 2.5e-4,
                 save_path="./seq2seq_example"):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.beam_search = beam_search
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm
        self.training_batchsize = batch_size
        self.lr_schedule = lr_schedule
        self.save_path = save_path
        
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)
        self.sw = SummaryWriter(log_dir=self.save_path)
        
        self._build_model()
        self._build_algorithm()
        self._prepare()
        
    def _build_model(self):
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.inputs_length = tf.placeholder(tf.int32, [None], name='inputs_length')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.targets_length = tf.placeholder(tf.int32, [None], name='targets_length')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        self.max_target_length = tf.reduce_max(self.targets_length, name='max_target_length')
        
        batch_size = tf.shape(self.inputs)[0]
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                'embedding', [self.vocab_size, self.embedding_size]
                )
        
        with tf.variable_scope('encoder'):
            with tf.device("/cpu:0"):
                self.encoder_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
            
            cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, output_keep_prob=self.keep_prob_placeholder
                )
            cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, output_keep_prob=self.keep_prob_placeholder
                )
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.encoder_inputs,
                sequence_length=self.inputs_length,
                dtype=tf.float32
                )
            self.encoder_state = encoder_fw_final_state + encoder_bw_final_state
            self.encoder_outputs = encoder_fw_outputs + encoder_bw_outputs
        
        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_cell, output_keep_prob=self.keep_prob_placeholder
                )
            
            if self.beam_search > 1: # inference
                start_tokens = tf.ones([batch_size], dtype=tf.int32, name='start_tokens') * (self.vocab_size - 2)
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                    self.encoder_outputs, multiplier=self.beam_search
                    )
                tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                    self.inputs_length, multiplier=self.beam_search
                    )
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size,
                    memory=tiled_encoder_outputs,
                    memory_sequence_length=tiled_sequence_length
                    )
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism
                    )
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_state, multiplier=self.beam_search
                    )
                tiled_decoder_initial_state = decoder_cell.zero_state(
                    batch_size=batch_size * self.beam_search, dtype=tf.float32
                    )
                decoder_initial_state = tiled_decoder_initial_state.clone(
                    cell_state=tiled_encoder_final_state
                    )
                self.decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=embedding,
                    start_tokens=start_tokens,
                    end_token=self.vocab_size - 1,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_search,
                    output_layer=tf.layers.Dense(self.vocab_size)
                    )
            else: # train
                ending = tf.strided_slice(
                    self.targets, [0, 0], [batch_size, -1], [1, 1]
                    )
                decoder_inputs = tf.concat(
                    [tf.fill([batch_size, 1], self.vocab_size - 2), ending], 1
                    )
                with tf.device("/cpu:0"):
                    self.decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_inputs,
                    sequence_length=self.targets_length
                    )
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size,
                    memory=self.encoder_outputs,
                    memory_sequence_length=self.inputs_length
                    )
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism
                    )
                decoder_initial_state = decoder_cell.zero_state(
                    batch_size=batch_size, dtype=tf.float32
                    )
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=self.encoder_state
                    )
                self.decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=tf.layers.Dense(self.vocab_size)
                    )
            
            self.decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder,
                maximum_iterations=self.max_target_length
                )
    
    def _build_algorithm(self):
        if self.beam_search == 1: # train
            self.moved_lr = tf.placeholder(tf.float32)
            self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)
            
            decoder_logits = tf.identity(self.decoder_outputs.rnn_output)
            sequence_mask = tf.sequence_mask(
                self.targets_length, self.max_target_length, dtype=tf.float32
                )
            self.total_loss = tf.contrib.seq2seq.sequence_loss(
                logits=decoder_logits,
                targets=self.targets,
                weights=sequence_mask
                )
            
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(
                zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())
        else: # inference
            self.preds = tf.identity(self.decoder_outputs.predicted_ids)
        
    def _prepare(self):
        self.saver = tf.train.Saver(max_to_keep=10)

        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.sess.run(tf.global_variables_initializer())
        self.load_model()
        
    def save_model(self):
        """Save model to `save_path`."""
        save_dir = os.path.join(self.save_path, "model")
        os.makedirs(save_dir, exist_ok=True)
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(save_dir, "model"),
            global_step,
            write_meta_graph=True
        )

    def load_model(self):
        """Load model from `save_path` if there exists."""
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.save_path, "model"))
        if latest_checkpoint:
            print("## Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("## New start!")
    
    def update(self, inputs, targets, inputs_length, targets_length, update_ratio):
        batch_datas = generator(
                inputs, targets, inputs_length, targets_length, self.training_batchsize
                )
        loss = 0
        step = 0
        for mini_inputs, mini_targets, mini_inputs_length, mini_targets_length in tqdm(batch_datas):
            step += 1
            fd = {
                self.inputs: mini_inputs,
                self.targets: mini_targets,
                self.inputs_length: mini_inputs_length,
                self.targets_length: mini_targets_length,
                self.keep_prob_placeholder: self.keep_prob,
                self.moved_lr: self.lr_schedule(update_ratio)
                }

            cur_loss, _ = self.sess.run(
                [self.total_loss, self.train_op],
                feed_dict=fd
                )
            loss += cur_loss
            global_step = self.sess.run(tf.train.get_global_step())
            self.sw.add_scalar(
                'loss',
                cur_loss,
                global_step=global_step)
        return loss / step
    
    def evaluate(self, eval_datas):
        step = 0
        loss = 0
        for mini_inputs, mini_targets, mini_inputs_length, mini_targets_length in tqdm(eval_datas):
            step += 1
            fd = {
                self.inputs: mini_inputs,
                self.targets: mini_targets,
                self.inputs_length: mini_inputs_length,
                self.targets_length: mini_targets_length,
                self.keep_prob_placeholder: 1.0
                }

            cur_loss = self.sess.run(self.total_loss, feed_dict=fd)
            loss += cur_loss
        return loss / step
    
    def inference(self, inputs, inputs_length, max_length):
        assert self.beam_search > 1
        fd = {
            self.inputs: inputs,
            self.inputs_length: inputs_length,
            self.targets_length: [max_length] * len(inputs),
            self.keep_prob_placeholder: 1.0
            }
        preds = self.sess.run([self.preds], feed_dict=fd)[0]
        return preds


if __name__ == "__main__":
    metadata, idx_q, idx_a = load_data(PATH='data/twitter/')
    train_ratio = 0.8
    
    n, p = idx_q.shape
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    hidden_size = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2
    
    inputs_length, targets_length = [], []
    for i in range(n):
        for j in range(1, p + 1):
            if idx_q[i][-j] != 0:
                inputs_length.append(p - j + 1)
                break
        for j in range(1, p + 1):
            if idx_a[i][-j] != 0:
                if j != 1:
                    idx_a[i][-j + 1] = end_id
                    targets_length.append(p - j + 2)
                else:
                    idx_a[i][-j] = end_id
                    targets_length.append(p)
                break
    
    inputs_length = np.asarray(inputs_length)
    targets_length = np.asarray(targets_length)
    train_size = int(train_ratio * n)
    test_size = n - train_size
    indexs = np.arange(n)
    np.random.seed(0)
    train_index = np.random.choice(indexs, size=train_size, replace=False)
    test_index = np.asarray(list(set(range(n)) - set(train_index)))
    
    train_inputs = idx_q[train_index]
    train_inputs_length = inputs_length[train_index]
    train_targets = idx_a[train_index]
    train_targets_length = targets_length[train_index]
    
    test_inputs = idx_q[test_index]
    test_inputs_length = inputs_length[test_index]
    test_targets = idx_a[test_index]
    test_targets_length = targets_length[test_index]
    
    eval_datas = generator(
        test_inputs, test_targets, test_inputs_length, test_targets_length, 64
        )
    
    total_updates = 100
    save_model_freq = 5
    
    vocab_size = src_vocab_size
    seq2seq = Seq2SeqModel(vocab_size, hidden_size, beam_search=1)
    
    epoch = 0
    while True:
        epoch += 1
        loss = seq2seq.update(
                train_inputs, train_targets, train_inputs_length, train_targets_length,
                min(0.9, epoch / total_updates))
        print(f'>>>>Train epoch: {epoch}, Loss: {loss}')
        if epoch % save_model_freq == 0:
            loss = seq2seq.evaluate(eval_datas)
            print(f'>>>>Test epoch: {epoch}, Loss: {loss}')
            seq2seq.save_model()
        
    '''
    seq2seq_ = Seq2SeqModel(vocab_size, hidden_size, beam_search=3)
    index = np.random.choice(test_index, size=3, replace=False)
    sample_inputs = idx_q[index]
    sample_targets = idx_a[index]
    sample_inputs_length = inputs_length[index]
    preds = seq2seq_.inference(sample_inputs, sample_inputs_length, 20)
    for i in range(3):
        print(f'input: {" ".join([idx2word[num] for num in sample_inputs[i] if num != 0])}')
        print(f'target: {" ".join([idx2word[num] for num in sample_targets[i] if num != 0])}')
        print(f'pred: {" ".join([idx2word[num] for num in preds[i, :, 0] if num != 0])}\n')
    '''
