import os

import sys
sys.path.append('./cleverhans')
from cleverhans.attacks_tf import fgm

import numpy as np
from tqdm import tqdm

from resnet.utils.adv_eval.fgm_target import fgm_target

def permute_labels(labels):
    true_labels = labels
    return (true_labels - 1) % 10

def gen_adv_examples(sess, model, data_iter, adv_attack):
    adv_examples = []
    labels = []
    target_labels = []

    iter_ = tqdm(data_iter)
    for batch in iter_:
        adv_targets = permute_labels(batch["label"])
        adv_examples.append(sess.run(adv_attack, {
            model.input: batch["img"],
            model.label: adv_targets
        }))
        labels.append(batch["label"])
        target_labels.append(adv_targets)

    return (np.array(adv_examples), np.array(labels), np.array(target_labels))

def save_adv_examples(sess, model, data_iter, logger, fgm_settings={np.inf: [0.1]}):
    '''
    fgm_settings should be a dictionary of the form { 'NORM' : 'ARRAY OF VALUES' }
    '''
    for norm in fgm_settings:
        for eps in fgm_settings[norm]:
            data_iter.reset()
            adv_examples, labels, _ = gen_adv_examples(sess, model, data_iter,
                                                       fgm(model.input, model.output, eps=eps, ord=norm))
            logger.log_adv_examples(adv_examples, labels, eps, norm)

    
    for norm in fgm_settings:
        for eps in fgm_settings[norm]:
            data_iter.reset()
            adv_examples, labels, targets = gen_adv_examples(sess, model, data_iter,
                                                             fgm_target(model.input, model.output, model.label, eps=eps, ord=norm))
            logger.log_adv_examples(adv_examples, labels, eps, norm, targets)

def load_adv_examples():
    pass


