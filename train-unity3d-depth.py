#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import os
import sys
import time
import random
import uuid
import argparse
import multiprocessing
import threading

#import cv2
from PIL import Image
import tensorflow as tf
import six
from six.moves import queue

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu


from tensorpack.RL import *
from simulator import *
import common
from common import (play_model, Evaluator, eval_model_multithread,
                    play_one_episode, play_n_episodes)

from unity3d_player import Unity3DPlayer

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 1
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 1
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 120
EVAL_EPISODE = 10
BATCH_SIZE = 30
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
SIMULATOR_PROC = None
PREDICTOR_THREAD_PER_GPU = 1
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None

SIMULATOR_IP_ADDRESS = '140.114.89.72'
EVAL_PORT = None
ACTION_SPACE_PORT = None

import cv2

DUMP_DIR = None

class RecordPlayer(ProxyPlayer):
    def __init__(self, pl, dump_dir):
        super(RecordPlayer, self).__init__(pl)
        self.dump_dir = dump_dir
        self.timestep = 0
        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #self.out = cv2.VideoWriter('%s.avi' % self.dump_dir, self.fourcc, 30.0, (640,480))

    def current_state(self):
        s = super(RecordPlayer, self).current_state()
        if self.timestep % 4 == 0:
            cv2.imwrite('%s/%06d.png' % (self.dump_dir, self.timestep), s)
        self.timestep += 1
        #self.out.write(s)
        return s

    def close(self):
        self.out.release()


class CloseablePlayer(ProxyPlayer):
    def __init__(self, pl, close_target):
        super(CloseablePlayer, self).__init__(pl)
        self.close_target = close_target
    def close(self):
        self.close_target.close()

def get_player(base_port, worker_id, viz=False, train=False, dumpdir=None, no_wrappers=False):
    # no_wrappers: for get raw player
    u3dpl = Unity3DPlayer(env_name=ENV_NAME, worker_id=worker_id, base_port=base_port, mode=train)
    if no_wrappers:
        return u3dpl
    repl = RecordPlayer(u3dpl, dumpdir)
    #pl = MapPlayerState(u3dpl, lambda img: cv2.resize(img, IMAGE_SIZE[::-1]))
    pl = MapPlayerState(repl, lambda img: np.expand_dims(np.array(Image.fromarray(img).
                                            resize(IMAGE_SIZE, resample=Image.BILINEAR), dtype=np.uint8)[:, :, 0], axis=-1))

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    else:
        pl = LimitLengthPlayer(pl, 5000)
    pl = CloseablePlayer(pl, [u3dpl, repl])
    return pl

def get_eval_player(worker_id, train):
    return get_player(base_port=EVAL_PORT, worker_id=worker_id, train=train, dumpdir=DUMP_DIR)

class MySimulatorWorker(SimulatorProcess): 
    def __init__(self, idx, pipe_c2s, pipe_s2c, base_port):
        super(MySimulatorWorker, self).__init__(idx, pipe_c2s, pipe_s2c)
        self.base_port = base_port

    def _build_player(self):
        return get_player(base_port=self.base_port, worker_id=self.idx, train=True, dumpdir=DUMP_DIR)


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward'),
                InputDesc(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, action, futurereward, action_prob = inputs
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   self.cost, tf.reduce_mean(importance, name='importance'))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        nr_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        def cb(outputs):
            try:
                distrib, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action]))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


def get_config():
    nr_gpu = get_nr_gpu()
    global PREDICTOR_THREAD
    if nr_gpu > 0:
        if nr_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    base_port = args.base_port
    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c, base_port=base_port) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    M = Model()
    master = MySimulatorMaster(namec2s, names2c, M, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        model=M,
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master)
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=10,
        tower=train_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--base_port', help='base port', required=True, type=int)
    parser.add_argument('--n_proc', help='n_proc', required=True, type=int)
    parser.add_argument('--env', help='env', default='Navigation')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'gen_submit'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--logdir', help='log directory', default=None)
    parser.add_argument('--dumpdir', help='dump directory', default=None)
    parser.add_argument('--episode', help='number of episode to eval', default=100, type=int)
    args = parser.parse_args()


    DUMP_DIR = args.dumpdir
    ENV_NAME = args.env
    SIMULATOR_PROC = args.n_proc
    EVAL_PORT = args.base_port + 1
    args.base_port += 1
    ACTION_SPACE_PORT = args.base_port + 1
    args.base_port += 1
    logger.info("Environment Name: {}".format(ENV_NAME))

    # port 9000 is used for get_action space
    tmp_player = get_player(base_port=ACTION_SPACE_PORT, worker_id=0, no_wrappers=True)

    NUM_ACTIONS = tmp_player.get_action_space().num_actions()
    tmp_player.close()
    del tmp_player 
    logger.info("Number of actions: {}".format(NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['policy'])
        if args.task == 'play':
            play_model(cfg, get_player(base_port=8000, worker_id=0, viz=0.01, dumpdir=DUMP_DIR))
        elif args.task == 'eval':
            eval_model_multithread(cfg, args.episode, get_eval_player)
        '''
        elif args.task == 'gen_submit':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                OfflinePredictor(cfg), args.episode)
            # gym.upload(output, api_key='xxx')
        '''
    else:
        if args.logdir:
            dirname = os.path.join('train_log', 'train-unity3d-{}-{}'.format(ENV_NAME, args.logdir))
        else:
            dirname = os.path.join('train_log', 'train-unity3d-{}'.format(ENV_NAME))
        logger.set_logger_dir(dirname)

        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = QueueInputTrainer if config.nr_tower == 1 else AsyncMultiGPUTrainer
        trainer(config).train()
