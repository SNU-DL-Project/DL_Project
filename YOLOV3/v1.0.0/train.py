from __future__ import division

import os
import tqdm

import torch
import torch.optim
from torchsummary import summary

from terminaltables import AsciiTable

from utils.dataload import create_train_data_loader, create_validation_data_loader
from utils.logger import Logger
from utils.utils import provide_determinism, to_cpu
from utils.loss import compute_yolo_loss
from utils.evaluate import yolo_evaluate

class Default_train:
    """
    1. Default_train
    상속만 가능
    자체로는 쓰이지 않는다.

    공통된 Train 과정
    (0) Data 불러오기 (train, val), model 불러오기
    (1) optimizer 정의, 업데이트
    (2) train(run)
    (3) progress log, print
    (4) parameter save (load는 model에서 함)
    (5) evaluate(val_data)
    """

    def __init__(self, model, data):
        """
        0. __init__
        :param model: model
        :param data: train, val data, classes
        """

        # [***다른 모듈에서 넘겨주어야 하는 값들]
        # model : YoloV3 모델(darknet.py)
        # data : YoloV3 train, val 데이터(dataload.py)

        # [**조절할 수 있는 값들]
        # seed : 추후에 재현을 위해, 고정시드를 제공. default는 랜덤시드(-1)
        # epochs : epochs
        # verbose : True면 모델 요약을 보여줌. False면 안 보여줌(default).
        # save_interval : model을 save하는 epoch 간격
        # evaluation_interval : val_data를 evaluation하는 epoch 간격

        # [안건드려도 되는 것]
        # n_cpu : cpu 수, 안건드려도 됨
        # logdir : logger가 저장되는 위치
        # multiscale_training : 몰라도 됨.

        self.model = model
        self.data = data
        self.train_dataloader
        self.validation_dataloader
        self.optimizer
        self.logger
        self.class_names

        self.verbose = False
        self.epochs = 100
        self.save_interval = 1
        self.evaluation_interval = 1
        self.seed = -1

        self.n_cpu = 8
        self.logdir='logs'
        self.multiscale_training=False

    # 자식 노드에서 구현
    def __optimizer(self):
        """
        (3) optimizer 정의
        :param params: model의 parameter
        :return: optimizer를 return
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        # 2가지 optimizer(adam, sgd)만 가능
        if (self.model.hyperparams['optimizer'] in [None, "adam"]):
            optimizer = torch.optim.Adam(params, lr=self.model.hyperparams['learning_rate'],
                                   weight_decay=self.model.hyperparams['decay'], )
        elif (self.model.hyperparams['optimizer'] == "sgd"):
            optimizer = torch.optim.SGD(params, lr=self.model.hyperparams['learning_rate'],
                                  weight_decay=self.model.hyperparams['decay'], momentum=self.model.hyperparams['momentum'])
        else:
            print("Unknown optimizer. Please choose between (adam, sgd).")

        return optimizer

    def __update_lr(self):
        pass

    # 이름 변경 필요
    def __logging(self):
        "자식 클래스에서 구현"
        pass

    def __evaluate(self):
        "자식 클래스에서 구현"
        pass

    def run(self):
        "자식 클래스에서 구현"
        pass



class YOLO_train(Default_train):
    def __init__(self, model, data):
        super(YOLO_train, self).__init__(model, data)
        self.logdir='logs_yolo'

        # [threshold 값들(건드릴 일 거의 없음)]
        # iou_thres = 0.5
        # conf_thres = 0.1
        # nms_thres = 0.5

        self.iou_thres = 0.5
        self.conf_thres = 0.1
        self.nms_thres = 0.5


    def __update_lr(self, batches_done):
        """
        (3) optimizer 업데이트
        :param params: model의 parameter
        :return: optimizer를 return
        """

        if batches_done % self.model.hyperparams['subdivisions'] == 0:
            # Adapt learning rate
            # Get learning rate defined in cfg
            lr = self.model.hyperparams['learning_rate']
            if batches_done < self.model.hyperparams['burn_in']:
                # Burn in
                lr *= (batches_done / self.model.hyperparams['burn_in'])
            else:
                # Set and parse the learning rate to the steps defined in the cfg
                for threshold, value in self.model.hyperparams['lr_steps']:
                    if batches_done > threshold:
                        lr *= value

            # Log the learning rate
            self.logger.scalar_summary("train/learning_rate", lr, batches_done)

            # Set learning rate
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            # Run optimizer
            self.optimizer.step()
            # Reset gradients
            self.optimizer.zero_grad()

    def __logging(self, batches_done, loss, loss_components):
        if self.verbose:
            print(AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])],
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table)

        # Tensorboard logging
        tensorboard_log = [
            ("train/iou_loss", float(loss_components[0])),
            ("train/obj_loss", float(loss_components[1])),
            ("train/class_loss", float(loss_components[2])),
            ("train/loss", to_cpu(loss).item())]
        self.logger.list_of_scalars_summary(tensorboard_log, batches_done)

    def __evaluate(self, epoch):
        """
        evaluate : 특정 epoch마다 val data의 metrics를 log한다.
        :param: epoch : epoch
        :return:
        """

        print("\n---- Evaluating Model ----")
        if epoch % self.evaluation_interval == 0:
            metrics_output = yolo_evaluate(
                self.model,
                self.validation_dataloader,
                self.class_names,
                img_size=self.model.hyperparams['height'],
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
                self.logger.list_of_scalars_summary(evaluation_metrics, epoch)

    def run(self, epochs):
        # epoch 수 설정
        self.epochs = epochs

        # 랜덤 시드 결정
        if self.seed != -1:
            provide_determinism(self.seed)

        # logger : tensorboard의 logger를 제공
        self.logger = Logger(self.logdir)

        # output과 checkpoints(save_interval 마다 모델이 저장) 폴더
        os.makedirs("output_yolo", exist_ok=True)
        os.makedirs("checkpoints_yolo", exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ######## 0.model,data ########
        # 모델과 데이터를 불러온다.
        # 이미 train 생성할때 불러옴...

        # verbose 값에 따라 model을 print
        if self.verbose:
            summary(self.model, input_size=(3, self.model.hyperparams['height'],
                                            self.model.hyperparams['height']))

        ######## 1.dataloader ########
        # dataload.py가 구현이 되어야 한다!!
        mini_batch_size = self.model.hyperparams['batch'] // self.model.hyperparams['subdivisions']
        self.train_dataloader = create_train_data_loader(train_path,
                        mini_batch_size,
                        self.model.hyperparams['height'],
                        self.n_cpu,
                        self.multiscale_training)
        self.validation_dataloader = create_validation_data_loader(
                        valid_path,
                        mini_batch_size,
                        self.model.hyperparams['height'],
                        self.n_cpu)


        ######## 2.optimizer ########
        self.optimizer = self.__optimizer()

        ######## 3.training ########
        for epoch in range(epochs):
            print("\n---- Training Model ----")
            self.model.train()

            for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(self.dataloader, desc=f"Training Epoch {epoch}")):
                batches_done = len(self.dataloader) * epoch + batch_i

                ######## 4.loss 계산 및 BP ########
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)
                outputs = self.model(imgs)

                loss, loss_components = compute_yolo_loss(outputs, targets, self.model)
                loss.backward()

                ######## 5.optimizer update ########
                # optimizer update, 자세히 몰라도 상관없음
                self.__update_lr(batches_done)

                ######## 6. log progress ########
                # verbose 출력 및 logging 작업 진행
                self.__logging(batches_done, loss, loss_components)

                ######## 7. save progress ########
                # 모델 파라미터 세이브 기능은 나중에 구현 예정
                # dataload.py가 먼저 구현이 되어야 한다.

                # model 출력 관련(model.seen은 모델 파라미터 세이브 관련)
                self.model.seen += imgs.size(0)
                '''
                # Save model to checkpoint file
                if epoch % args.checkpoint_interval == 0:
                    checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
                    print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
                    torch.save(model.state_dict(), checkpoint_path)
                '''

                ######## 8.Evaluate ########
                # val data를 통해 모델을 평가하는 기능
                self.__evaluate(epoch)


# GAN울 쓰면 필요해질수도....
class GAN_train(Default_train):
    pass