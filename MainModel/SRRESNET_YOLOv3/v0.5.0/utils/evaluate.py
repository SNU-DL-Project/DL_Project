from __future__ import division

import tqdm
import math
import numpy as np
from terminaltables import AsciiTable
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .srutils import ssim
from .yoloutils import ap_per_class, get_yolo_batch_statistics, non_max_suppression, xywh2xyxy
from .augutils import sr_downsample

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


def print_srres_eval_stats(metrics_output, verbose):
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

def srres_evaluate(model_srres, dataloader, verbose):
    model_srres.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    real_img_list=[]; fake_img_list=[]; low_img_list=[]; mse_list=[]; ssim_list=[];psnr_list=[];

    for _, real_imgs,_ in tqdm.tqdm(dataloader, desc="Validating"):

        batch_size = real_imgs.size(0)
        model_srres.hyperparams['lr_height']=104 # size 정의
        lr_imgs = sr_downsample(real_imgs, noise=False, down_size=int(model_srres.hyperparams['lr_height']))
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        real_imgs = real_imgs.to(device, non_blocking=True)


        with torch.no_grad():
            fake_imgs = model_srres(lr_imgs)

        valing_results['batch_sizes'] += batch_size

        batch_mse = ((fake_imgs - real_imgs) ** 2).mean()
        mse_step = batch_mse * batch_size
        valing_results['mse'] += mse_step;
        batch_ssim = ssim(fake_imgs, real_imgs).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * math.log10((real_imgs.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']));
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes'];

        #parameter 전달용
        fake_img_list.append(fake_imgs)
        low_img_list.append(lr_imgs)
        real_img_list.append(real_imgs)
        ssim_list.append(valing_results['ssim'])
        psnr_list.append(valing_results['psnr'])
        mse_list.append(round(mse_step.item(),8))


    #MSE, PSNR, SSIM
    metrics_output = [float(valing_results['mse']/valing_results['batch_sizes']),
                      float(valing_results['psnr']), float(valing_results['ssim'])]
    print_srres_eval_stats(metrics_output, verbose)
    return metrics_output, fake_img_list, real_img_list, low_img_list , psnr_list, mse_list, ssim_list