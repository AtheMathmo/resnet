#!/usr/bin/env python
"""
Trains a CNN on ImageNet.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_exp.py --model           [MODEL NAME]                     \
                      --config          [CONFIG FILE]                    \
                      --id              [EXPERIMENT ID]                  \
                      --logs            [LOGS FOLDER]                    \
                      --results         [SAVE FOLDER]                    \
                      --restore                                          \
                      --norestore                                        \
                      --max_num_steps   [MAX NUM OF STEPS FOR THIS RUN]  \
                      --num_gpu         [NUMBER OF GPU]                  \
                      --num_pass        [NUMBER OF FW/BW PASS]

Flags:
  --model: Model type. Available options are:
       1) resnet-50
       2) resnet-101
  --id: Experiment ID, optional for new experiment.
  --config: Not using the pre-defined configs above, specify the JSON file
    that contains model configurations.
  --logs: Path to logs folder, default is ./logs/default.
  --results: Path to save folder, default is ./results/imagenet.
  --restore: Whether or not to restore checkpoint. Checkpoint should be 
    present in [SAVE FOLDER]/[EXPERIMENT ID] folder.
  --max_num_steps: Maximum number of steps for this training session.
  --num_gpu: Number of GPU to perform data parallelism.
  --num_pass: Number of forward and backward passes to average gradients.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm

from resnet.configs.imagenet_exp_config import get_config, get_config_from_json
from resnet.data import get_dataset
from resnet.models import (ResNetModel, MultiTowerModel,
                           MultiPassMultiTowerModel)
from resnet.utils import (ExperimentLogger, FixedLearnRateScheduler,
                          ExponentialLearnRateScheduler)
from resnet.utils import logger, gen_id

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file")
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_string("model", "resnet-50", "Model name")
flags.DEFINE_bool("restore", False, "Restore checkpoint")
flags.DEFINE_integer("max_num_steps", -1, "Maximum number of steps")
flags.DEFINE_integer("num_gpu", 4, "Number of GPUs")
flags.DEFINE_integer("num_pass", 1, "Number of forward-backwad passes")
FLAGS = flags.FLAGS

DATASET = "imagenet"


def _get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return get_config_from_json(FLAGS.config)
  else:
    return get_config(DATASET, FLAGS.model)


def get_model(config, num_replica, num_pass, is_training):
  if num_replica > 1:
    if num_pass > 1:
      return MultiPassMultiTowerModel(
          config,
          ResNetModel,
          num_replica=num_replica,
          is_training=is_training,
          num_passes=num_pass)
    else:
      return MultiTowerModel(
          config, ResNetModel, num_replica=num_replica, is_training=is_training)
  elif num_replica == 1:
    return ResNetModel(config, is_training=is_training)
  else:
    raise Exception("Unacceptable number of replica: {}".format(num_replica))


def train_step(sess, model, batch):
  """Train step."""
  ce = model.train_step(sess, batch["img"], batch["label"])
  return ce


def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(config.to_json())
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def train_model(exp_id, config, train_iter, save_folder=None, logs_folder=None):
  """Trains a CIFAR model.

  Args:
    exp_id: String. Experiment ID.
    config: Config object
    train_data: Dataset object

  Returns:
    acc: Final test accuracy
  """
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)

  # Initializes variables.
  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)

    # Builds models.
    log.info("Building models")
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        m = get_model(
            config,
            num_replica=FLAGS.num_gpu,
            num_pass=FLAGS.num_pass,
            is_training=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver = tf.train.Saver(max_to_keep=None)  ### Keep all checkpoints here!
      if FLAGS.restore:
        log.info("Restore checkpoint \"{}\"".format(save_folder))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
      else:
        sess.run(tf.global_variables_initializer())

      max_train_iter = config.max_train_iter
      niter_start = int(m.global_step.eval())

      # Add upper bound to the number of steps.
      if FLAGS.max_num_steps > 0:
        max_train_iter = min(max_train_iter, niter_start + FLAGS.max_num_steps)

      # Set up learning rate schedule.
      if config.lr_scheduler == "fixed":
        lr_scheduler = FixedLearnRateScheduler(
            sess,
            m,
            config.base_learn_rate,
            config.lr_decay_steps,
            lr_list=config.lr_list)
      elif config.lr_scheduler == "exponential":
        lr_scheduler = ExponentialLearnRateScheduler(
            sess, m, config.base_learn_rate, config.lr_decay_offset,
            config.max_train_iter, config.final_learn_rate,
            config.lr_decay_interval)
      else:
        raise Exception("Unknown learning rate scheduler {}".format(
            config.lr_scheduler))

      for niter in tqdm(range(niter_start, config.max_train_iter), desc=exp_id):
        lr_scheduler.step(niter)
        ce = train_step(sess, m, train_iter.next())

        if (niter + 1) % config.disp_iter == 0 or niter == 0:
          exp_logger.log_train_ce(niter, ce)

        if (niter + 1) % config.save_iter == 0 or niter == 0:
          if save_folder is not None:
            save(sess, saver, m.global_step, config, save_folder)
          exp_logger.log_learn_rate(niter, m.lr.eval())


def main():
  # Loads parammeters.
  config = _get_config()

  if FLAGS.id is None:
    exp_id = "exp_" + DATASET + "_" + FLAGS.model
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id

  if FLAGS.results is not None:
    save_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.results, exp_id)))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
  else:
    save_folder = None

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(
      DATASET,
      "train",
      batch_size=config.batch_size,
      preprocessor=config.preprocessor)

  # Trains a model.
  train_model(
      exp_id,
      config,
      train_data,
      save_folder=save_folder,
      logs_folder=logs_folder)


if __name__ == "__main__":
  main()
