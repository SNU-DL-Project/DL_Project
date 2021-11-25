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
from utils.loss import compute_yolo_loss, compute_srres_loss
from utils.evaluate import yolo_evaluate, srres_evaluate
from utils.augutils import sr_downsample

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
        # logger : 하위 클래스에서 구현
        # multiscale_training : 몰라도 됨.
        # train_dataloader : default를 공유
        # validation_dataloader : default를 공유

        # [하위 클래스에서 구현할 변수]
        # model : SR/YOLO 모두 다름
        # optimizer : 마찬가지

        self.data_config = data_config
        self.train_dataloader = None
        self.validation_dataloader = None

        self.hyperparams = {'batch': 4, 'subdivisions': 1, 'height': 416, 'width': 416}
        self.verbose = True
        self.epochs = 100
        self.save_interval = 1
        self.evaluation_interval = 1
        self.seed = -1

        self.n_cpu = 4
        self.trainingdir = 'training'
        self.logger = None
        self.multiscale_training=False

    def _train_init(self, epochs):
        # epoch 수 설정
        self.epochs = epochs

        # 랜덤 시드 결정
        if self.seed != -1:
            provide_determinism(self.seed)

        # output과 checkpoints(save_interval 마다 모델이 저장) 폴더
        os.makedirs(self.trainingdir + "/output", exist_ok=True)
        os.makedirs(self.trainingdir + "/checkpoints", exist_ok=True)

        # logger
        self.logger = Logger(self.trainingdir + "/logs")

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

        self.trainingdir='training/yolo'
        self.hyperparams.update(self.model_yolo.hyperparams)

        self.optimizer_yolo = None
        self.class_names = load_classes(self.data_config["names"])

        # [threshold 값들(건드릴 일 거의 없음)]
        # iou_thres = 0.5
        # conf_thres = 0.1
        # nms_thres = 0.5

        self.iou_thres = 0.5
        self.conf_thres = 0.1
        self.nms_thres = 0.5

    def _train_one_step_yolo(self, imgs, targets):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        outputs = self.model_yolo(imgs)

        loss, loss_components = compute_yolo_loss(outputs, targets, self.model_yolo)
        loss.backward()
        return loss, loss_components

    def _optim_init_yolo(self):
        """
            (3) optimizer 초기화 yolo

            :param params: model의 parameter
            :return: optimizer를 return
        """
        params = [p for p in self.model_yolo.parameters() if p.requires_grad]
        optimizer = None
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
            self.logger.scalar_summary("train/learning_rate", lr, batches_done)

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
        self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

    def _saving_yolo(self, epoch):
        if epoch % self.save_interval == 0:
            checkpoint_path = self.trainingdir + f"/checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"\n---- Saving YOLO checkpoint to: '{checkpoint_path}' ----")
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
                        ("validation/f1", f1.mean()),
                        ]
                for it in range(len(AP)):
                    metric_location = "validation/AP_" + str(it)
                    evaluation_metrics.append((metric_location, AP[it]))
                self.logger.list_of_scalars_summary(evaluation_metrics, epoch)

    def yolo_save(self, yolo_save_path, cutoff_yolo=None):
        if cutoff_yolo:
            self.model_yolo.save_darknet_weights(yolo_save_path, cutoff_yolo)
        else:
            self.model_yolo.save_darknet_weights(yolo_save_path)

    def run(self, epochs):
        ######## 0.train 초기화 ########
        self._train_init(epochs)

        ######## 1.model,data ########
        # verbose 값에 따라 model을 print
        if self.verbose:
            summary(self.model_yolo, input_size=(3, self.hyperparams['height'], self.hyperparams['height']))

        ######## 2.dataloader ########
        self._dataloader_init()

        ######## 3.optimizer ########
        self._optim_init_yolo()

        ######## 4.training ########
        for epoch in range(epochs):
            print("\n---- Training YOLO Model ----")
            self.model_yolo.train()

            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i

                ######## 5.FP,BP ########
                loss, loss_components = self._train_one_step_yolo(imgs, targets)

                ######## 6.optimizer update ########
                # optimizer update, 자세히 몰라도 상관없음
                self._optim_update_yolo(batches_done)

                ######## 7.log progress ########
                # verbose 출력 및 logging 작업 진행
                self._logging_yolo(batches_done, loss, loss_components)

                self.model_yolo.seen += imgs.size(0)

            ######## 8.save progress ########
            # Save model to checkpoint file
            self._saving_yolo(epoch)

            ######## 9.Evaluate ########
            # val data를 통해 모델을 평가하는 기능
            self._evaluate_yolo(epoch)


class SRRES_train(Default_train):
    def __init__(self, model_srres, data_config, upscale=4):
        super(SRRES_train, self).__init__(data_config)
        self.model_srres = model_srres

        self.trainingdir = 'training/srres'
        self.upscale = upscale
        self.hyperparams.update(self.model_srres.hyperparams)
        self.hyperparams['upscale'] = self.upscale
        self.hyperparams['lr_height'] = int(self.hyperparams['height'] // self.upscale)
        self.hyperparams['lr_width'] = int(self.hyperparams['width'] // self.upscale)
        self.model_srres.hyperparams['upscale'] = self.hyperparams['upscale']
        self.model_srres.hyperparams['lr_height'] = self.hyperparams['lr_height']
        self.model_srres.hyperparams['lr_width'] = self.hyperparams['lr_width']

        self.optimizer_srres = None

    def _train_one_step_srres(self, real_imgs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lr_imgs = sr_downsample(real_imgs, down_size=self.hyperparams['lr_height'])
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        real_imgs = real_imgs.to(device, non_blocking=True)

        lr_imgs = Variable(lr_imgs)
        lr_imgs.to(device)
        real_imgs = Variable(real_imgs)
        real_imgs.to(device)

        self.model_srres.zero_grad()
        # The two lines below are added to prevent runetime error in Google Colab
        fake_imgs = self.model_srres(lr_imgs)
        loss, loss_score = compute_srres_loss(fake_imgs, real_imgs)
        loss.backward()
        #fake_imgs = self.model_srres(lr_imgs)

        '''
        # 이미지 확인
        import matplotlib.pyplot as pp
        A = fake_imgs.detach().cpu().numpy()
        B = real_imgs.detach().cpu().numpy()
        C = lr_imgs.detach().cpu().numpy()
        A = A.reshape(3, 416, 416)
        B = B.reshape(3, 416, 416)
        C = C.reshape(3, 208, 208)
        A = A.swapaxes(0, 1)
        A = A.swapaxes(1, 2)
        B = B.swapaxes(0, 1)
        B = B.swapaxes(1, 2)
        C = C.swapaxes(0, 1)
        C = C.swapaxes(1, 2)
        pp.imshow(A)
        pp.show()
        pp.imshow(B)
        pp.show()
        pp.imshow(C)
        pp.show()
        '''

        return loss, loss_score

    def _optim_init_srres(self):
        params = [p for p in self.model_srres.parameters() if p.requires_grad]
        optimizer = None
        # 2가지 optimizer(adam, sgd)만 가능
        if (self.model_srres.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = torch.optim.Adam(params, lr=self.model_srres.hyperparams['learning_rate'],
                                         weight_decay=self.model_srres.hyperparams['decay'], )
        elif (self.model_srres.hyperparams['optimizer'] == "sgd"):
            optimizer = torch.optim.SGD(params, lr=self.model_srres.hyperparams['learning_rate'],
                                        weight_decay=self.model_srres.hyperparams['decay'],
                                        momentum=self.model_srres.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")

        self.optimizer_srres = optimizer

    def _optim_update_srres(self, batches_done):
        if batches_done % self.model_srres.hyperparams['subdivisions'] == 0:
            # Adapt learning rate
            # Get learning rate defined in cfg
            lr = self.model_srres.hyperparams['learning_rate']
            if batches_done < self.model_srres.hyperparams['burn_in']:
                # Burn in
                lr *= (batches_done / self.model_srres.hyperparams['burn_in'])
            else:
                # Set and parse the learning rate to the steps defined in the cfg
                for threshold, value in self.model_srres.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value

            # Log the learning rate
            self.logger.scalar_summary("train/learning_rate", lr, batches_done)

            # Set learning rate
            for g in self.optimizer_srres.param_groups:
                g['lr'] = lr

            # Run optimizer
            self.optimizer_srres.step()
            # Reset gradients
            self.optimizer_srres.zero_grad()

    def _logging_srres(self, batches_done, loss, loss_score):
        """
            (4) _logging_srgan : 특정 epoch마다 val data의 metrics를 log한다.
            :param: batches_done : batches_done
            :param: loss : loss
            :param: loss_component : loss_component
        """
        # score 확률분포도 넣을까 생각중
        if self.verbose:
            print('\n', AsciiTable(
                [
                    ["Type", "Value"],
                    ["Img loss", float(loss_score[0])],
                    ["Tv loss", float(loss_score[1])],
                    ["Loss", float(loss)],
                ]).table, sep='')

        # Tensorboard logging
        tensorboard_log = [
            ("train/img_loss", float(loss_score[0])),
            ("train/tv_loss", float(loss_score[1])),
            ("train/loss", float(loss)),
        ]
        self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

    def _saving_srres(self, epoch):
        if epoch % self.save_interval == 0:
            checkpoint_path = self.trainingdir + f"/checkpoints/srres_ckpt_{epoch}.pth"
            print(f"\n---- Saving SRRES checkpoint to: '{checkpoint_path}' ----")
            torch.save(self.model_srres.state_dict(), checkpoint_path)

    def _evaluate_srres(self, epoch):
        """
        (4) _evaluate_srgan : 특정 epoch마다 val data의 metrics를 log한다.
        :param: epoch : epoch
        """
        print("\n---- Evaluating SRRES Model ----")
        if epoch % self.evaluation_interval == 0:
            metrics_output = srres_evaluate(
                self.model_srres,
                self.validation_dataloader,
                verbose=self.verbose
            )

            if metrics_output is not None:
                mse, psnr, ssim = metrics_output
                evaluation_metrics = [
                    ("validation/mse", mse),
                    ("validation/psnr", psnr),
                    ("validation/ssim", ssim), ]
                self.logger.list_of_scalars_summary(evaluation_metrics, epoch)

    def srres_save(self, srres_save_path, cutoff_srres=None):
        if cutoff_srres:
            self.model_srres.save_srres_weights(srres_save_path, cutoff_srres)
        else:
            self.model_srres.save_srres_weights(srres_save_path)

    def run(self, epochs):
        ######## 0.train 초기화 ########
        self._train_init(epochs)

        ######## 1.model,data ########
        # verbose 값에 따라 model을 print
        if self.verbose:
            summary(self.model_srres, input_size=(3, self.hyperparams['lr_height'], self.hyperparams['lr_height']))

        ######## 2.dataloader ########
        self._dataloader_init()

        ######## 3.optimizer ########
        self._optim_init_srres()

        ######## 4.training ########
        for epoch in range(epochs):
            print("\n---- Training SRRES Model ----")
            self.model_srres.train()

            for batch_i, (_, real_imgs, _) in enumerate(tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i
                ######## 5.FP,BP ########
                loss, loss_score = self._train_one_step_srres(real_imgs)

                ######## 6.optimizer update ########
                self._optim_update_srres(batches_done)

                ######## 7. log progress ########
                # verbose 출력 및 logging 작업 진행
                self._logging_srres(batches_done, loss, loss_score)

                self.model_srres.seen += real_imgs.size(0)

            ######## 7. save progress ########
            # Save model to checkpoint file
            self._saving_srres(epoch)

            ######## 8.Evaluate ########
            # val data를 통해 모델을 평가하는 기능
            self._evaluate_srres(epoch)



'''
class SRRES_YOLO_train(Default_train):
    def __init__(self, model_srres, model_yolo, data_config):
        super(SRRES_YOLO_train, self).__init__(model_srres=model_srres, data_config=data_config)
        super(SRRES_YOLO_train, self).__init__(model_yolo=model_yolo, data_config=data_config)

        self.trainingdir = 'training/srres_yolo'

    def _train_one_step(self):
        pass
        # 구현 중

    def _optim_init_srgan_yolo(self):
        self._optim_init_srgan()
        self._optim_init_yolo()

    def _optim_update_srgan_yolo(self, batches_done):
        """
            (3) optimizer 업데이트
            :param params: model의 parameter
            :return: optimizer를 return
        """
        self._optim_update_srgan(batches_done)
        self._optim_update_yolo(batches_done)

    def _logging_srgan_yolo(self, batches_done, srgan_loss_score, yolo_loss, yolo_loss_components):
        """
            (4) _logging_srgan : 특정 epoch마다 val data의 metrics를 log한다.
            :param: batches_done : batches_done
            :param: loss : loss
            :param: loss_component : loss_component
        """
        self._logging_srgan(batches_done, srgan_loss_score)
        self._logging_yolo(batches_done, yolo_loss, yolo_loss_components)

    def _saving_srgan_yolo(self, epoch):
        if epoch % self.save_interval == 0:
            checkpoint_path_G = f"training/srgan_yolo/checkpoints/srganG_ckpt_{epoch}.pth"
            checkpoint_path_D = f"training/srgan_yolo/checkpoints/srganD_ckpt_{epoch}.pth"
            checkpoint_path_yolo = f"training/srgan_yolo/checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"\n---- Saving checkpoint to: '{checkpoint_path_yolo}' ----")
            torch.save(self.model_G.state_dict(), checkpoint_path_G)
            torch.save(self.model_D.state_dict(), checkpoint_path_D)
            torch.save(self.model_yolo.state_dict(), checkpoint_path_yolo)

    def _evaluate_srgan_yolo(self, epoch):
        """
        (4) _evaluate_srgan : 특정 epoch마다 val data의 metrics를 log한다.
        :param: epoch : epoch
        """
        self._evaluate_srgan(epoch)
        self._evaluate_yolo(epoch)

    def run(self, epochs):
        # epoch 수 설정
        self.epochs = epochs

        # 랜덤 시드 결정
        if self.seed != -1:
            provide_determinism(self.seed)

        # output과 checkpoints(save_interval 마다 모델이 저장) 폴더
        os.makedirs("training/srgan_yolo/output", exist_ok=True)
        os.makedirs("training/srgan_yolo/checkpoints", exist_ok=True)

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
            print("\n---- Training SRGAN Model ----")
            self.model_D.train()
            self.model_G.train()

            for batch_i, (_, real_imgs, _) in enumerate(
                    tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i

                ######## 4.input,output,gt ########
                print(real_imgs.size())
                lr_imgs = srgan_downsample(real_imgs, self.hyperparams['batch'])
                print(real_imgs.size())
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

                loss_score = [D_loss.item(), G_loss.item(), real_out.item(), fake_out.item()]

                ######## 6.optimizer update ########
                self._optim_update_srgan(batches_done)

                ######## 7. log progress ########
                # verbose 출력 및 logging 작업 진행
                self._logging_srgan(batches_done, loss_score)

                self.model_G.seen += real_imgs.size(0)
                self.model_D.seen += real_imgs.size(0)

            ######## 7. save progress ########
            # Save model to checkpoint file
            self._saving_srgan_yolo(epoch)

            ######## 8.Evaluate ########
            # val data를 통해 모델을 평가하는 기능
            self._evaluate_srgan_yolo(epoch)


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

            for batch_i, (_, imgs, targets) in enumerate(
                    tqdm.tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.train_dataloader) * epoch + batch_i

                ######## 4.FP, BP ########
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)
                outputs = self.model_yolo(imgs)

                loss, loss_components = compute_yolo_loss(outputs, targets, self.model_yolo)
                loss.backward()
                # loss, loss_components = self._propagation_yolo(imgs, targets)

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