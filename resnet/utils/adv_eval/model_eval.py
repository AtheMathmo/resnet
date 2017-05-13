import tensorflow as tf
import sys
import numpy as np

sys.path.append('../../../cleverhans')
from cleverhans.attacks_tf import fgsm
import matplotlib.pyplot as plt

from universal_pert import universal_perturbation

def permute_labels(labels):
    true_labels = np.argmax(labels, axis=1)
    adv_labels = (true_labels - 1) % 10
    one_hot = np.zeros(labels.shape)
    one_hot[np.arange(labels.shape[0]), adv_labels] = 1.0
    return one_hot

def fgs_eval(sess, model, data_iter, fgs_eps):
    '''
    Returns (untargeted_fgs_acc, targeted_fgs_acc)
    '''
    untarget_num_correct = 0.0
    untarget_count = 0
    for batch in data_iter:
        # Try to perturb with cleverhans (untargeted)
        perturbed_imgs_fgs = sess.run(
            fgsm(model.input, model.output, fgs_eps, 0.0, 1.0), {
                model.input: batch["img"]
        })
        y = sess.run(model.output, {
            model.input: perturbed_imgs_fgs
        })
        pred_label = np.argmax(y, axis=1)
        untarget_num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
        untarget_count += pred_label.size
    untargeted_pred_acc = (untarget_num_correct / untarget_count)

    # Here we generate targeted adversarial attacks
    # target_perturbed_imgs_fgs = model.perturb_inputs_fgs(fgs_eps, imgs, target_labels)
    # untargeted_pred_acc = model.predictive_accuracy(perturbed_imgs_fgs, labels)
    # targeted_pred_acc = model.predictive_accuracy(target_perturbed_imgs_fgs, target_labels)

    return untargeted_pred_acc

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

def adv_eval(sess, model, data_iter, fgs_eps=0.1):
    untarget_fgs_acc = fgs_eval(sess, model, data_iter, fgs_eps)

    # Pick some image and plot its true vs the cleverhans adversary
    # img_idx = 0
    # all_im = np.concatenate((correct_pred_imgs[img_idx].reshape((32,32,3)),
    #                          perturbed_imgs_fgs[img_idx].reshape((32,32,3))),
    #                          axis=1)
    # plt.imshow(all_im, cmap='gray')
    # plt.show()

    print("Untargeted FGS pred acc: ", untarget_fgs_acc)