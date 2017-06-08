from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm

from resnet.configs.cifar_exp_config import get_config, get_config_from_json
from resnet.data import get_dataset
from resnet.data.adv_examples import save_adv_examples
from resnet.models import ResNetModel
from resnet.utils import ExperimentLogger, AdvLogger, logger
from resnet.utils.adv_eval.model_eval import adv_eval

flags = tf.flags
flags.DEFINE_string("id", None, "eExperiment ID")
flags.DEFINE_string("dataset", "cifar-10", "Dataset name.")
flags.DEFINE_string("results", "./results/cifar", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_string("config", None, "Custom JSON config file")
flags.DEFINE_string("model", "resnet-32", "Model type.")
flags.DEFINE_string("mode", "eval", "Run mode. 'eval' or 'save'")
FLAGS = tf.flags.FLAGS
log = logger.get()

FGM_SETTINGS = {
    np.inf: [0.0,  1e-2, 0.1, 1.0, 3.0, 5.0, 10.0, 25.0, 50.0],
    1: [5.0, 10.0, 30.0, 70.0, 100.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0],
    2: [0.1, 0.3, 0.5, 0.7, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
}

def _get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return get_config_from_json(FLAGS.config)
  else:
    return get_config(FLAGS.dataset, FLAGS.model)

def get_model(config):
    with tf.name_scope("Valid"):
        with tf.variable_scope("Model"):
            mvalid = ResNetModel(config, is_training=False)
    return mvalid

def evaluate(sess, model, data_iter):
    """Runs evaluation."""
    num_correct = 0.0
    count = 0
    iter_ = tqdm(data_iter)
    for batch in iter_:
        y = model.infer_step(sess, batch["img"])
        pred_label = np.argmax(y, axis=1)
        num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
        count += pred_label.size
    acc = (num_correct / count)
    return acc

def eval_model(config, train_data, test_data, save_folder, logs_folder=None):
  log.info("Config: {}".format(config.__dict__))

  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)
    exp_logger = ExperimentLogger(logs_folder)
    adv_logger = AdvLogger(logs_folder)

    # Builds models.
    log.info("Building models")
    mvalid = get_model(config)

    # # A hack to load compatible models.
    # variables = tf.global_variables()
    # names = map(lambda x: x.name, variables)
    # names = map(lambda x: x.replace("Model/", "Model/Towers/"), names)
    # names = map(lambda x: x.replace(":0", ""), names)
    # var_dict = dict(zip(names, variables))

    # Initializes variables.
    with tf.Session() as sess:
      # saver = tf.train.Saver(var_dict)
      saver = tf.train.Saver()
      ckpt = tf.train.latest_checkpoint(save_folder)
      # log.fatal(ckpt)
      saver.restore(sess, ckpt)
      train_acc = evaluate(sess, mvalid, train_data)
      val_acc = evaluate(sess, mvalid, test_data)

      # evaluate adversarial robustness
      test_data.reset()
      adv_eval(sess, mvalid, test_data, logger=adv_logger)

      niter = int(ckpt.split("-")[-1])
      exp_logger.log_train_acc(niter, train_acc)
      exp_logger.log_valid_acc(niter, val_acc)
    return val_acc

def only_adv_eval(config, train_data, test_data, save_folder, logs_folder=None):
    log.info("Config: {}".format(config.__dict__))
    adv_logger = AdvLogger(logs_folder)

    with tf.Graph().as_default():
        np.random.seed(0)
        tf.set_random_seed(1234)

        # Builds models.
        log.info("Building models")
        mvalid = get_model(config)

        # # A hack to load compatible models.
        # variables = tf.global_variables()
        # names = map(lambda x: x.name, variables)
        # names = map(lambda x: x.replace("Model/", "Model/Towers/"), names)
        # names = map(lambda x: x.replace(":0", ""), names)
        # var_dict = dict(zip(names, variables))

        # Initializes variables.
        with tf.Session() as sess:
            # saver = tf.train.Saver(var_dict)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(save_folder)
            # log.fatal(ckpt)
            saver.restore(sess, ckpt)
            adv_eval(sess, mvalid, train_data, test_data, FGM_SETTINGS, logger=adv_logger)

def gen_and_save_adv_examples(config, test_data, save_folder, logs_folder=None):
    log.info("Config: {}".format(config.__dict__))
    adv_logger = AdvLogger(logs_folder)

    with tf.Graph().as_default():
        np.random.seed(0)
        tf.set_random_seed(1234)

        # Builds models.
        log.info("Building models")
        mvalid = get_model(config)

        # Initializes variables.
        with tf.Session() as sess:
            # saver = tf.train.Saver(var_dict)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(save_folder)
            # log.fatal(ckpt)
            saver.restore(sess, ckpt)
            save_adv_examples(sess, mvalid, test_data, adv_logger, fgm_settings=FGM_SETTINGS)

def main():
  config = _get_config()
  exp_id = FLAGS.id

  save_folder = os.path.realpath(
      os.path.abspath(os.path.join(FLAGS.results, exp_id)))

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
      FLAGS.dataset,
      "train",
      cycle=False,
      data_aug=False,
      prefetch=False)
  test_data = get_dataset(
      FLAGS.dataset,
      "test",
      cycle=False,
      data_aug=False,
      prefetch=False)

  # Evaluates a model.
  #eval_model(config, train_data, test_data, save_folder, logs_folder)
  if FLAGS.mode.lower() == 'eval':
    only_adv_eval(config, train_data, test_data, save_folder, logs_folder)
  elif FLAGS.mode.lower() == 'save':
      gen_and_save_adv_examples(config, test_data, save_folder, logs_folder)


if __name__ == "__main__":
  main()