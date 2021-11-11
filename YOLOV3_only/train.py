from __future__ import division

import os
import tqdm

import torch
import torch.optim
from torchsummary import summary
from torch.autograd import Variable

from terminaltables import AsciiTable

from utils.load import load_classes
from utils.dataloader import create_train_data_loader, create_validation_data_loader
from utils.logger import Logger
from utils.utils import provide_determinism, to_cpu
from utils.loss import compute_yolo_loss, GeneratorLoss
from utils.evaluate import yolo_evaluate, srgan_evaluate
from utils.augutils import srgan_downsample

from abc import *
class Default_train(metaclass=ABCMeta):
    """
    1. Default_train
    상속만 가능, 자체로는 쓰이지 않는다. (추상 클래스)

    Train 과정
    (0) Data 불러오기 (train, val), model 불러오기
    (1) optimizer 정의, 업데이트
    (2) train(run)
    (3) progress log, print
    (4) parameter save (load는 model에서 함)
    (5) evaluate(val_data)
    """
    def __init__(self, data_config):
        """
        0. __init__
        :param data: train, val data, classes
        """

        # [***다른 모듈에서 넘겨주어야 하는 변수]
        # data : SR/Yolo train, val 데이터(dataloader.py)

        # [**조절할 수 있는 변수]
        # hyperparams : train할 때 필요한 여러 정보들(model.hyperparams)
        # seed : 추후에 재현을 위해, 고정시드를 제공. default는 랜덤시드(-1)
        # epochs : epochs
        # verbose : True면 모델 요약을 보여줌. False면 안 보여줌(default).
        # save_interval : model을 save하는 epoch 간격
        # evaluation_interval : val_data를 evaluation하는 epoch 간격

        # [안건드려도 되는 변수]
        # n_cpu : cpu 수, 안건드려도 됨
        # logdir : logger가 저장되는 위치
        # multiscale_training : 몰라도 됨.
        # train_dataloader : default를 공유
        # validation_dataloader : default를 공유

        # [하위 클래스에서 구현할 변수]
        # model : SR/YOLO 모두 다름
        # optimizer : 마찬가지
        # logger : 마찬가지

        self.data_config = data_config
        self.train_dataloader = None
        self.validation_dataloader = None

        self.hyperparams = {'batch': 4, 'subdivisions': 1, 'height': 416, 'width': 416}
        self.verbose = True
        self.epochs = 100
        self.save_interval = 10
        self.evaluation_interval = 10
        self.seed = -1

        self.n_cpu = 4
        self.logdir='training/logs'
        self.multiscale_training=False

    def _dataloader_init(self):
        train_path = self.data_config["train"]
        valid_path = self.data_config["valid"]
        mini_batch_size = self.hyperparams['batch'] // self.hyperparams['subdivisions']
        self.train_dataloader = create_train_data_loader(train_path,
                                                         mini_batch_size,
                                                         self.hyperparams['height'],
                                                         self.n_cpu,
                                                         self.multiscale_training)
        self.validation_dataloader = create_validation_data_loader(
            valid_path,
            mini_batch_size,
            self.hyperparams['height'],
            self.n_cpu)

    @abstractmethod
    def run(self, epochs):
        "자식 클래스에서 구현"
        pass

class YOLO_train(Default_train):
    def __init__(self, model, data_config):
        super(YOLO_train, self).__init__(data_config)
        self.model_yolo = model

        self.logdir='training/yolo/logs'
        self.hyperparams.update(self.model_yolo.hyperparams)

        self.optimizer_yolo = None
        self.logger_yolo = Logger(self.logdir)
        self.class_names = load_classes(self.data_config["names"])

        # [threshold 값들(건드릴 일 거의 없음)]
        # iou_thres = 0.5
        # conf_thres = 0.1
        # nms_thres = 0.5

        self.iou_thres = 0.5
        self.conf_thres = 0.1
        self.nms_thres = 0.5

    def _optim_init_yolo(self):
        """
            (3) optimizer 초기화 yolo

            :param params: model의 parameter
            :return: optimizer를 return
        """
        params = [p for p in self.model_yolo.parameters() if p.requires_grad]

        # 2가지 optimizer(adam, sgd)만 가능
        if (self.model_yolo.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = torch.optim.Adam(params, lr=self.model_yolo.hyperparams['learning_rate'],
                                         weight_decay=self.model_yolo.hyperparams['decay'], )
        elif (self.model_yolo.hyperparams['optimizer'] == "sgd"):
            optimizer = torch.optim.SGD(params, lr=self.model_yolo.hyperparams['learning_rate'],
                                        weight_decay=self.model_yolo.hyperparams['decay'],
                                        momentum=self.model_yolo.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")

        self.optimizer_yolo = optimizer

    def _optim_update_yolo(self, batches_done):
        """
            (3) optimizer 업데이트
            :param params: model의 parameter
            :return: optimizer를 return
        """

        if batches_done % self.model_yolo.hyperparams['subdivisions'] == 0:
            # Adapt learning rate
            # Get learning rate defined in cfg
            lr = self.model_yolo.hyperparams['learning_rate']
            if batches_done < self.model_yolo.hyperparams['burn_in']:
                # Burn in
                lr *= (batches_done / self.model_yolo.hyperparams['burn_in'])
            else:
                # Set and parse the learning rate to the steps defined in the cfg
                for threshold, value in self.model_yolo.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value

            # Log the learning rate
            self.logger_yolo.scalar_summary("train/learning_rate", lr, batches_done)

            # Set learning rate
            for g in self.optimizer_yolo.param_groups:
                g['lr'] = lr

            # Run optimizer
            self.optimizer_yolo.step()
            # Reset gradients
            self.optimizer_yolo.zero_grad()

    def _logging_yolo(self, batches_done, loss, loss_components):
        """
            (4) _logging_yolo : 특정 epoch마다 val data의 metrics를 log한다.
            :param: batches_done : batches_done
            :param: loss : loss
            :param: loss_component : loss_component
        """
        if self.verbose:
            print('\n', AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])],
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table, sep='')

        # Tensorboard logging
        tensorboard_log = [
            ("train/iou_loss", float(loss_components[0])),
            ("train/obj_loss", float(loss_components[1])),
            ("train/class_loss", float(loss_components[2])),
            ("train/loss", to_cpu(loss).item())]
        self.logger_yolo.list_of_scalars_summary(tensorboard_log, batches_done)

    def _saving_yolo(self, epoch):
        if epoch % self.save_interval == 0:
            checkpoint_path = f"training/yolo/checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"\n---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(self.model_yolo.state_dict(), checkpoint_path)

    def _evaluate_yolo(self, epoch):
        """
        (4) _evaluate_yolo : 특정 epoch마다 val data의 metrics를 log한다.
        :param: epoch : epoch
        """

        print("\n---- Evaluating YOLO Model ----")
        if epoch % self.evaluation_interval == 0:
            metrics_output = yolo_evaluate(
                self.model_yolo,
                self.validation_dataloader,
                self.class_names,
                img_size=self.model_yolo.hyperparams['height'],
                iou_thres=self.iou_thres,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                verbose=self.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                        ("validation/precision", precision.mean()),
                        ("validation/recall", recall.mean()),
                        ("validation/mAP", AP.mean()),
                        ("validation/f1", f1.mean())]
                self.logger_yolo.list_of_scalars_summary(evaluation_metrics, epoch)

    def run(self, epochs):
        # epoch 수 설정
        self.epochs = epochs

        # 랜덤 시드 결정
        if self.seed != -1:
            provide_determinism(self.seed)

        # output과 checkpoints(save_interval 마다 모델이 저장) 폴더
        os.makedirs("training/yolo/output", exist_ok=True)
        os.makedirs("training/yolo/checkpoints", exist_ok=True)

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ######## 0.model,data ########
        # 모델과 데이터를 불러온다.(model, data_config)
        # init할 때 불러와져 있다.
        # verbose 값에 따라 model을 print
        if self.verbose:
            summary(self.model_yolo, input_size=(3, self.hyperparams['height'], self.hyperparams['height']))

        ######## 1.dataloader ########
        self._dataloader_init()

        ######## 2.optimizer ########
        self._optim_init_yolo()

        ######## 3.training ########
        for epoch in range(epochs):
            print("\n---- Training YOLO Model ----")
            self.model_yolo.train()

            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i

                ######## 4.loss 계산 및 BP ########
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)
                outputs = self.model_yolo(imgs)

                loss, loss_components = compute_yolo_loss(outputs, targets, self.model_yolo)
                loss.backward()

                ######## 5.optimizer update ########
                # optimizer update, 자세히 몰라도 상관없음
                self._optim_update_yolo(batches_done)

                ######## 6. log progress ########
                # verbose 출력 및 logging 작업 진행
                self._logging_yolo(batches_done, loss, loss_components)

                self.model_yolo.seen += imgs.size(0)

            ######## 7. save progress ########
            # Save model to checkpoint file
            self._saving_yolo(epoch)

            ######## 8.Evaluate ########
            # val data를 통해 모델을 평가하는 기능
            self._evaluate_yolo(epoch)

'''
class SRGAN_train(Default_train):
    def __init__(self, model_G, model_D, data_config):
        super(SRGAN_train, self).__init__(data_config)
        self.model_G = model_G
        self.model_D = model_D
        
        self.logdir='training/srgan/logs'
        self.upscale = 2
        self.hyperparams.update(self.model_G.hyperparams)
        self.hyperparams['upscale'] = self.upscale
        self.hyperparams['lr_height'] = int(self.hyperparams['height'] / self.upscale)
        self.hyperparams['lr_width'] = int(self.hyperparams['width'] / self.upscale)

        self.optimizer_G = None
        self.optimizer_D = None
        self.logger_srgan = Logger(self.logdir)

    def _optim_init_srgan(self):        
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters())
        self.optimizer_D = torch.optim.Adam(self.model_D.parameters())

    def _optim_update_srgan(self, batches_done):
        """
            (3) optimizer 업데이트
            :param params: model의 parameter
            :return: optimizer를 return
        """

        if batches_done % self.hyperparams['subdivisions'] == 0:
            # Log the learning rate
            lr_D = self.optimizer_D.param_groups[0]['lr']
            lr_G = self.optimizer_G.param_groups[0]['lr']
            self.logger_srgan.scalar_summary("train/learning_rate_D", lr_D, batches_done)
            self.logger_srgan.scalar_summary("train/learning_rate_G", lr_G, batches_done)

            # Run optimizer
            self.optimizer_D.step()
            self.optimizer_G.step()
            # Reset gradients
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()

    def _logging_srgan(self, batches_done, loss_components):
        """
            (4) _logging_srgan : 특정 epoch마다 val data의 metrics를 log한다.
            :param: batches_done : batches_done
            :param: loss : loss
            :param: loss_component : loss_component
        """
        if self.verbose:
            print('\n', AsciiTable(
                [
                    ["Type", "Value"],
                    ["D loss", float(loss_components[0])],
                    ["G loss", float(loss_components[1])],
                    ["D score", float(loss_components[2])],
                    ["G score", float(loss_components[3])],
                ]).table, sep='')

        # Tensorboard logging
        tensorboard_log = [
            ("train/D_loss", float(loss_components[0])),
            ("train/G_loss", float(loss_components[1])),
            ("train/D_score", float(loss_components[2])),
            ("train/G_score", float(loss_components[3]))]
        self.logger_srgan.list_of_scalars_summary(tensorboard_log, batches_done)

    def _saving_srgan(self, epoch):
        if epoch % self.save_interval == 0:
            checkpoint_path_G = f"training/srgan/checkpoints/srganG_ckpt_{epoch}.pth"
            checkpoint_path_D = f"training/srgan/checkpoints/srganD_ckpt_{epoch}.pth"
            print(f"\n---- Saving checkpoint to: '{checkpoint_path_G}' ----")
            torch.save(self.model_G.state_dict(), checkpoint_path_G)
            torch.save(self.model_D.state_dict(), checkpoint_path_D)

    def _evaluate_srgan(self, epoch):
        """
        (4) _evaluate_srgan : 특정 epoch마다 val data의 metrics를 log한다.
        :param: epoch : epoch
        """
        print("\n---- Evaluating SRGAN Model ----")
        if epoch % self.evaluation_interval == 0:
            metrics_output = srgan_evaluate(
                self.model_D,
                self.model_G,
                self.validation_dataloader,
                verbose=self.verbose
            )

            if metrics_output is not None:
                mse, psnr, ssim = metrics_output
                evaluation_metrics = [
                        ("validation/mse", mse),
                        ("validation/psnr", psnr),
                        ("validation/ssim", ssim),]
                self.logger_srgan.list_of_scalars_summary(evaluation_metrics, epoch)

    def run(self, epochs):
        # epoch 수 설정
        self.epochs = epochs

        # 랜덤 시드 결정
        if self.seed != -1:
            provide_determinism(self.seed)

        # output과 checkpoints(save_interval 마다 모델이 저장) 폴더
        os.makedirs("training/srgan/output", exist_ok=True)
        os.makedirs("training/srgan/checkpoints", exist_ok=True)

        # device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # criterion
        generator_criterion = GeneratorLoss().to(device)

        ######## 0.model,data ########
        # 모델과 데이터를 불러온다.(model, data_config)
        # init할 때 불러와져 있다.
        # verbose 값에 따라 model을 print
        if self.verbose:
            summary(self.model_D, input_size=(3, self.hyperparams['height'], self.hyperparams['height']))
            summary(self.model_G, input_size=(3, self.hyperparams['lr_height'], self.hyperparams['lr_height']))

        ######## 1.dataloader ########
        self._dataloader_init()

        ######## 2.optimizer ########
        self._optim_init_srgan()

        ######## 3.training ########
        for epoch in range(epochs):
            print("\n---- Training Model ----")
            self.model_D.train()
            self.model_G.train()

            for batch_i, (_, real_imgs, _) in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i

                ######## 4.input,output,gt ########
                lr_imgs = srgan_downsample(real_imgs)
                lr_imgs = lr_imgs.to(device, non_blocking=True)
                real_imgs = real_imgs.to(device, non_blocking=True)

                lr_imgs = Variable(lr_imgs)
                lr_imgs.to(device)
                real_imgs = Variable(real_imgs)
                real_imgs.to(device)

                ######## 5.Backprop ########
                # D update :  maximize D(x)-1-D(G(z))
                fake_imgs = self.model_G(lr_imgs)
                self.model_D.zero_grad()
                real_out = self.model_D(real_imgs).mean()
                fake_out = self.model_D(fake_imgs).mean()
                D_loss = 1 - real_out + fake_out
                D_loss.backward(retain_graph=True)

                # G update : minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                self.model_G.zero_grad()
                # The two lines below are added to prevent runetime error in Google Colab
                fake_imgs = self.model_G(lr_imgs)
                fake_out = self.model_D(fake_imgs).mean()
                G_loss = generator_criterion(fake_out, fake_imgs, real_imgs)
                G_loss.backward()
                fake_imgs = self.model_G(lr_imgs)
                fake_out = self.model_D(fake_imgs).mean()

                loss_components = [D_loss.item(), G_loss.item(), real_out.item(), fake_out.item()]

                ######## 6.optimizer update ########
                self._optim_update_srgan(batches_done)

                ######## 7. log progress ########
                # verbose 출력 및 logging 작업 진행
                self._logging_srgan(batches_done, loss_components)

                self.model_G.seen += lr_imgs.size(0)
                self.model_D.seen += lr_imgs.size(0)

            ######## 7. save progress ########
            # Save model to checkpoint file
            self._saving_srgan(epoch)

            ######## 8.Evaluate ########
            # val data를 통해 모델을 평가하는 기능
            self._evaluate_srgan(epoch)



            #eval
            self.model_G.eval()
            out_path = 'training_results/SRF_' + str(self.UPSCALE_FACTOR) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for val_lr, val_hr_restore, val_hr in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = self.model_G(lr)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10(
                        (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                         display_transform()(sr.data.cpu().squeeze(0))])
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1

            #logging
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

            if epoch % 10 == 0 and epoch != 0:
                out_path = 'statistics/'
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                          'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))
                data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
                
'''
