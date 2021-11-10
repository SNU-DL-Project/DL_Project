from __future__ import division

import tqdm
import math
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from .srganutils import ssim
from .yoloutils import ap_per_class, get_yolo_batch_statistics, non_max_suppression, xywh2xyxy
from .augutils import srgan_downsample

def print_yolo_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print("\n", AsciiTable(ap_table).table, sep='')
        print(f"\n---- mAP {AP.mean():.5f} ----")
    else:
        print("\n---- mAP not measured (no detections found by model) ----")

def yolo_evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_yolo_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("\n---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print_yolo_eval_stats(metrics_output, class_names, verbose)

    return metrics_output

def print_srgan_eval_stats(metrics_output, verbose):
    if metrics_output is not None:
        mse, psnr, ssim = metrics_output
        if verbose:
            print('\n', AsciiTable(
                [
                    ["Type", "Value"],
                    ["mse", mse],
                    ["psnr", psnr],
                    ["ssim", ssim],
                ]).table, sep='')

        print(f"\n---- PSNR {psnr:.5f}, SSIM {ssim:.5f} ----")
    else:
        print("\n---- PSNR, SSIM not measured (no detections found by model) ----")

def srgan_evaluate(model_D, model_G, dataloader, verbose):
    ##편의를 위한 기능. 넣어야할까?
    ##gan image save 기능
    ##precision-recall 그래프 기능
    ##yolo val detection image 기능

    model_D.eval()
    model_G.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    for _, real_imgs, _ in tqdm.tqdm(dataloader, desc="Validating"):
        batch_size = real_imgs.size(0)
        lr_imgs = srgan_downsample(real_imgs, noise=False)
        lr_imgs.to(device)
        real_imgs.to(device)

        with torch.no_grad():
            fake_imgs = model_G(lr_imgs)

        batch_mse = ((fake_imgs - real_imgs) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = ssim(fake_imgs, real_imgs).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * math.log10((real_imgs.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

    if len(valing_results['mse']) == 0:  # No detections over whole validation set.
        print("\n---- No detections over whole validation set ----")
        return None

    #MSE, PSNR, SSIM
    metrics_output = [valing_results['mse']/valing_results['batch_sizes'], valing_results['psnr'], valing_results['ssim']]
    print_srgan_eval_stats(metrics_output, verbose)
    return metrics_output

