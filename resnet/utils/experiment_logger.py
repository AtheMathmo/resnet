from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import os
import sys

from resnet.utils import logger

import numpy as np

log = logger.get()


class ExperimentLogger():
  """Writes experimental logs to CSV file."""

  def __init__(self, logs_folder):
    """Initialize files."""
    self._write_to_csv = logs_folder is not None

    if self._write_to_csv:
      if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder)

      catalog_file = os.path.join(logs_folder, "catalog")

      with open(catalog_file, "w") as f:
        f.write("filename,type,name\n")

      with open(catalog_file, "a") as f:
        f.write("{},plain,{}\n".format("cmd.txt", "Commands"))

      with open(os.path.join(logs_folder, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))

      with open(catalog_file, "a") as f:
        f.write("train_ce.csv,csv,Train Loss (Cross Entropy)\n")
        f.write("train_acc.csv,csv,Train Accuracy\n")
        f.write("valid_acc.csv,csv,Validation Accuracy\n")
        f.write("learn_rate.csv,csv,Learning Rate\n")

      self.train_file_name = os.path.join(logs_folder, "train_ce.csv")
      if not os.path.exists(self.train_file_name):
        with open(self.train_file_name, "w") as f:
          f.write("step,time,ce\n")

      self.trainval_file_name = os.path.join(logs_folder, "train_acc.csv")
      if not os.path.exists(self.trainval_file_name):
        with open(self.trainval_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.val_file_name = os.path.join(logs_folder, "valid_acc.csv")
      if not os.path.exists(self.val_file_name):
        with open(self.val_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.lr_file_name = os.path.join(logs_folder, "learn_rate.csv")
      if not os.path.exists(self.lr_file_name):
        with open(self.lr_file_name, "w") as f:
          f.write("step,time,lr\n")

  def log_train_ce(self, niter, ce):
    """Writes training CE."""
    log.info("Train Step = {:06d} || CE loss = {:.4e}".format(niter + 1, ce))
    if self._write_to_csv:
      with open(self.train_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), ce))

  def log_train_acc(self, niter, acc):
    """Writes training accuracy."""
    log.info("Train accuracy = {:.3f}".format(acc * 100))
    if self._write_to_csv:
      with open(self.trainval_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_valid_acc(self, niter, acc):
    """Writes validation accuracy."""
    log.info("Valid accuracy = {:.3f}".format(acc * 100))
    if self._write_to_csv:
      with open(self.val_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_learn_rate(self, niter, lr):
    """Writes validation accuracy."""
    if self._write_to_csv:
      with open(self.lr_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), lr))

class AdvLogger():

    def __init__(self, logs_folder):
      self._write_to_csv = logs_folder is not None

      if self._write_to_csv:
          self.adv_atk_filename = os.path.join(logs_folder, 'adv_atk_stats.csv')
          self.adv_examples_folder = os.path.join(logs_folder, 'adv_examples')

          if not os.path.isdir(self.adv_examples_folder):
              os.makedirs(self.adv_examples_folder)

          if not os.path.exists(self.adv_atk_filename):
              with open(self.adv_atk_filename, "w") as f:
                  f.write("eps,untarget_acc,target_acc,target_atk_success,norm\n")

    def log_adv_stats(self, norm, eps, untarget_acc, target_acc, target_atk_success):
        with open(self.adv_atk_filename, "a") as f:
          f.write("{},{},{},{},{}\n".format(
              eps, untarget_acc, target_acc, target_atk_success, norm
          ))

    def log_adv_examples(self, adv_examples, true_labels, fgs_eps, fgs_norm, target_labels=None):
        if target_labels is not None:
          targets_filename = os.path.join(self.adv_examples_folder,
                                          'targets_t_{}_{}'.format(fgs_norm, fgs_eps))
          np.save(targets_filename, target_labels)
          ex_filename = os.path.join(self.adv_examples_folder,
                                     'adv_examples_t_{}_{}'.format(fgs_norm, fgs_eps))
          labels_filename = os.path.join(self.adv_examples_folder,
                                         'labels_t_{}_{}'.format(fgs_norm, fgs_eps))
        else:
          ex_filename = os.path.join(self.adv_examples_folder,
                                     'adv_examples_{}_{}'.format(fgs_norm, fgs_eps))
          labels_filename = os.path.join(self.adv_examples_folder,
                                         'labels_{}_{}'.format(fgs_norm, fgs_eps))
        np.save(ex_filename, adv_examples)
        np.save(labels_filename, true_labels)
