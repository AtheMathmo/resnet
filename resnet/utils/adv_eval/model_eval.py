import tensorflow as tf
import sys
import numpy as np

sys.path.append('./cleverhans')
from cleverhans.attacks_tf import fgsm
import matplotlib.pyplot as plt

from tqdm import tqdm

from universal_pert import universal_perturbation

def permute_labels(labels):
    true_labels = labels #np.argmax(labels, axis=1)
    return (true_labels - 1) % 10
    # one_hot = np.zeros(labels.shape)
    # one_hot[np.arange(labels.shape[0]), adv_labels] = 1.0
    # return one_hot

def target_fgs_attack(model, fgs_eps, clip_min, clip_max):
    adv_ce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model.label, logits=model.logits))
    input_grad = tf.gradients(adv_ce, model.input)[0]
    return model.input - fgs_eps * tf.sign(input_grad)

def fgs_eval(sess, model, data_iter, fgs_eps):
    '''
    Returns (untargeted_fgs_acc, targeted_fgs_acc)
    '''
    untarget_num_correct = 0.0
    target_num_correct = 0.0
    target_atk_success = 0.0
    total_count = 0
    iter_ = tqdm(data_iter)
    fgsm_attack = fgsm(model.input, model.output, fgs_eps)
    targeted_fgsm_attack = target_fgs_attack(model, fgs_eps, 0.0, 255.0)

    for batch in iter_:
        target_labels = permute_labels(batch["label"])
        # Try to perturb with cleverhans (untargeted)
        perturbed_imgs_fgs = sess.run(fgsm_attack, {
            model.input: batch["img"]
        })
        targeted_fgs_imgs = sess.run(targeted_fgsm_attack, {
            model.input: batch["img"],
            model.label: target_labels
        })
        y_untarget = sess.run(model.output, {
            model.input: perturbed_imgs_fgs
        })
        y_targeted = sess.run(model.output, {
            model.input: targeted_fgs_imgs
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

def eval_jacobian_things(sess, model, imgs, targets):
    logit_jac_norm = sess.run(model.true_jac_norm, {
        model.input: imgs
    })

    dbp_norm = sess.run(model.dbp_loss_norm, {
        model.input: imgs,
        model.label: targets
    })

    return logit_jac_norm, dbp_norm

def adv_eval(sess, model, data_iter, eps_range=[0.1], logger=None):
    for fgs_eps in eps_range:
        data_iter.reset()
        untarget_fgs_acc, targeted_pred_acc, targeted_success_rate = fgs_eval(sess, model, data_iter, fgs_eps)

        if logger is not None:
            logger.log_adv_stats(fgs_eps, untarget_fgs_acc, targeted_pred_acc, targeted_success_rate)

        print("------- For eps = {} -------".format(fgs_eps))
        print("Untargeted FGS pred acc: ", untarget_fgs_acc)
        print("Targeted FGS pred acc:", targeted_pred_acc)
        print("Targeted FGS attack success:", targeted_success_rate)

    data_iter.reset()
    res = data_iter.get_fn(np.arange(100))
    print(eval_jacobian_things(sess, model, res["img"], res["label"]))
