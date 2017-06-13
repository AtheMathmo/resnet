import os

import sys
sys.path.append('./cleverhans')
from cleverhans.attacks_tf import fgm

import numpy as np
from tqdm import tqdm

from resnet.utils.adv_eval.fgm_target import fgm_target
from resnet.data.dataset import AdvExamplesDataset
from resnet.data.get_dataset import get_iter

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

    return (np.concatenate(adv_examples), np.concatenate(labels), np.concatenate(target_labels))

def save_adv_examples(sess, model, data_iter, save_folder, fgm_settings={np.inf: [0.1]}):
    '''
    fgm_settings should be a dictionary of the form { 'NORM' : 'ARRAY OF VALUES' }
    '''
    examples_folder = os.path.join(save_folder, 'adv_examples')
    if not os.path.isdir(examples_folder):
        os.makedirs(examples_folder)

    for norm in fgm_settings:
        for eps in fgm_settings[norm]:
            data_iter.reset()
            adv_examples, labels, _ = gen_adv_examples(sess, model, data_iter,
                                                       fgm(model.input, model.output, eps=eps, ord=norm))
            _save_adv_examples(examples_folder, adv_examples, labels, eps, norm)

    for norm in fgm_settings:
        for eps in fgm_settings[norm]:
            data_iter.reset()
            adv_examples, labels, targets = gen_adv_examples(sess, model, data_iter,
                                                             fgm_target(model.input, model.output, model.label, eps=eps, ord=norm))
            _save_adv_examples(examples_folder, adv_examples, labels, eps, norm, targets)


def _save_adv_examples(folder, adv_examples, true_labels, fgs_eps, fgs_norm, target_labels=None):
    if target_labels is not None:
        targets_filename = os.path.join(folder,
                                        'targets_t_{}_{}'.format(fgs_norm, fgs_eps))
        np.save(targets_filename, target_labels)
        ex_filename = os.path.join(folder,
                                    'adv_examples_t_{}_{}'.format(fgs_norm, fgs_eps))
        labels_filename = os.path.join(folder,
                                        'labels_t_{}_{}'.format(fgs_norm, fgs_eps))
    else:
        ex_filename = os.path.join(folder,
                                    'adv_examples_{}_{}'.format(fgs_norm, fgs_eps))
        labels_filename = os.path.join(folder,
                                        'labels_{}_{}'.format(fgs_norm, fgs_eps))
    np.save(ex_filename, adv_examples)
    np.save(labels_filename, true_labels)


def load_adv_examples(save_folder, fgm_eps, fgm_norm, targeted=False):
    folder = os.path.join(save_folder, 'adv_examples')
    if targeted:
        targets_path = os.path.join(folder,
                                    'targets_t_{}_{}.npy'.format(fgm_norm, fgm_eps))
        examples_path = os.path.join(folder,
                                     'adv_examples_t_{}_{}.npy'.format(fgm_norm, fgm_eps))
        labels_filename = os.path.join(folder,
                                       'labels_t_{}_{}.npy'.format(fgm_norm, fgm_eps))
        targets = np.load(targets_path)
        examples = np.load(examples_path)
        labels = np.load(labels_filename)
    else:
        examples_path = os.path.join(folder,
                                     'adv_examples_{}_{}.npy'.format(fgm_norm, fgm_eps))
        labels_filename = os.path.join(folder,
                                       'labels_{}_{}.npy'.format(fgm_norm, fgm_eps))
        examples = np.load(examples_path)
        labels = np.load(labels_filename)
        targets = None

    dataset = AdvExamplesDataset(examples, labels, targets)
    return get_iter(dataset)
