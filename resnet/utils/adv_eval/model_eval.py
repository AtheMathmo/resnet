import tensorflow as tf
import sys
import numpy as np

sys.path.append('./cleverhans')
from cleverhans.attacks_tf import fgm
import matplotlib.pyplot as plt

from tqdm import tqdm

from universal_pert import universal_perturbation
from fgm_target import fgm_target

def permute_labels(labels):
    true_labels = labels #np.argmax(labels, axis=1)
    return (true_labels - 1) % 10
    # one_hot = np.zeros(labels.shape)
    # one_hot[np.arange(labels.shape[0]), adv_labels] = 1.0
    # return one_hot

def fgs_eval(sess, model, data_iter, fgm_eps, norm=np.inf, logger=None):
    '''
    Returns (untargeted_fgs_acc, targeted_fgs_acc, targeted_atk_success_rate)
    '''
    untarget_num_correct = 0.0
    target_num_correct = 0.0
    target_atk_success = 0.0
    total_count = 0
    iter_ = tqdm(data_iter)
    fgm_attack = fgm(model.input, model.output, eps=fgm_eps, ord=norm)
    targeted_fgm_attack = fgm_target(model.input, model.output, model.label, eps=fgm_eps, ord=norm)

    for batch in iter_:
        target_labels = permute_labels(batch["label"])
        # Try to perturb with cleverhans (untargeted)
        perturbed_imgs_fgm = sess.run(fgm_attack, {
            model.input: batch["img"]
        })
        targeted_fgm_imgs = sess.run(targeted_fgm_attack, {
            model.input: batch["img"],
            model.label: target_labels
        })
        y_untarget = sess.run(model.output, {
            model.input: perturbed_imgs_fgm
        })
        y_targeted = sess.run(model.output, {
            model.input: targeted_fgm_imgs
        })

        untarget_pred_label = np.argmax(y_untarget, axis=1)
        target_pred_label = np.argmax(y_targeted, axis=1)
        untarget_num_correct += np.sum(np.equal(untarget_pred_label, batch["label"]).astype(float))
        target_num_correct += np.sum(np.equal(target_pred_label, batch["label"]).astype(float))
        target_atk_success += np.sum(np.equal(target_pred_label, target_labels).astype(float))
        total_count += untarget_pred_label.size
    untargeted_pred_acc = (untarget_num_correct / total_count)
    targeted_pred_acc = (target_num_correct / total_count)
    targeted_success_rate = (target_atk_success / total_count)

    # Here we generate targeted adversarial attacks
    # target_perturbed_imgs_fgs = model.perturb_inputs_fgs(fgs_eps, imgs, target_labels)
    # untargeted_pred_acc = model.predictive_accuracy(perturbed_imgs_fgs, labels)
    # targeted_pred_acc = model.predictive_accuracy(target_perturbed_imgs_fgs, target_labels)

    logger.log_adv_stats(norm, fgm_eps, untargeted_pred_acc, targeted_pred_acc, targeted_success_rate)
    return (untargeted_pred_acc, targeted_pred_acc, targeted_success_rate)

def deepfool_eval(model, imgs, delta=0.2):
    def f(image_input):
        return model.sess.run(model.output, {
            model.input: image_input
        })

    def f_grad(image_input, inds):
        return model.sess.run(model.jacobians, {
            model._input: image_input
        })

    v = universal_perturbation(imgs, f, f_grad, delta=delta)

def eval_jacobian_things(sess, model, dataset, imgs, targets, logger=None):
    logit_jac_norm = sess.run(model.true_jac_norm, {
        model.input: imgs
    })

    dbp_norm = sess.run(model.dbp_loss_norm, {
        model.input: imgs,
        model.label: targets
    })

    if logger is not None:
        logger.log_jacobian(dataset, logit_jac_norm, dbp_norm)

    return logit_jac_norm, dbp_norm

def adv_eval(sess, model, train_data, test_data, fgm_settings={np.inf: [0.1]}, logger=None):
    for norm in fgm_settings:
        for eps in fgm_settings[norm]:
            test_data.reset()
            untarget_fgs_acc, targeted_pred_acc, targeted_success_rate = fgs_eval(sess, model, test_data, eps, norm, logger=logger)

            print("------- For norm = {}, eps = {} -------".format(norm, eps))
            print("Untargeted FGS pred acc: ", untarget_fgs_acc)
            print("Targeted FGS pred acc:", targeted_pred_acc)
            print("Targeted FGS attack success:", targeted_success_rate)

    # Evaluate the model Jacobian
    test_data.reset()
    test_samples = test_data.get_fn(np.arange(100))

    train_data.reset()
    train_samples = train_data.get_fn(np.arange(100))

    eval_jacobian_things(sess, model, 'test', test_samples["img"], test_samples["label"])
    eval_jacobian_things(sess, model, 'train', train_samples["img"], train_samples["label"])

