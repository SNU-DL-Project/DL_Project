##### 작성 중 입니다.
------------
# NOTICE

##### https://github.com/eriklindernoren/PyTorch-YOLOv3 에 있는 코드를 보고 구현중입니다.
##### 처음부터 코드를 구현하는게 아니라, 위의 코드를 목적에 맞게 수정하는 작업을 진행 중입니다. (그리고 가독성도 높이고 있습니다.)
##### code를 그대로 가져다가 약간 변형시킨 정도라 어느정도까지 copy의 범위일지는 조교님께 물어보아야 할 것 같습니다.
##### Augmentation은 YOLOV3 내부에 포함하지 않고, 따로 진행하여야할 것 같습니다.
------------
# YOLOV3 v.1.0.0

## 코드 구성
### (layer 0)
> main.py : API

### (layer 1)
> darknet.py : weight 로드받아 darknet을 모델링한다. 세이브 기능도 존재.  
> train.py : 모델 및 데이터를 입력받아 train 한다. loss, weight를 기록, 저장  
> test.py : inference test

### (layer 2)
> dataload.py : 외부 데이터(yolo .cfg 파일, weight 파일, dataset 등) 로드, 세이브  
> layers.py : yolo layer 정의, configuration 파일을 layer로 변환  
> loss.py : yolo loss를 계산  
> logger.py : 파이토치 log를 print, 저장하는 기능  
> evaluate.py : evaluation을 진행. test, train에 사용 

### (layer 3)
> yoloutils.py : yolov3에 필요한 기타 기능들 (IoU 계산, 좌표변환 등)  
> utils.py : 일반적으로 필요로하는 기타 기능들 (to_CPU 등)  

## layer 0
### main.py
main API

## layer 1
### darknet.py
yolov3의 메인 네트워크 모듈
1. class Darknet : Yolov3 layer 구조 (<- dataload.py, layers.py)
+ forward 
+ parameter save 기능 (추후 구현 예정)
### train.py
train을 진행하는 모듈
1. class Default_train : train template, (model, data)를 입력받음 (<- dataload.py)
+ def __optimizer : optimizer 정의
+ def __update_lr : learning rate update
+ def __logging : loss, iou 등을 기록 및 print (<- logger.py)
+ def __evaluate : val에 대한 evaluate 및 기록 (<- evaluate.py)
+ def run : 외부에서 train을 실행 <- loss.py
2. class YOLO_train : yolov3 train, 위의 class를 상속받아 yolov3에 맞게 메소드 오버라이딩
3. class GAN_train : gan train, 위의 class를 상속받아 gan에 맞게 메소드 오버라이딩
### test.py
test를 진행하는 모듈 (구현 필요X)

## layer 2
### dataload.py
data load/save와 관련된 모듈
1. def cfg_to_block : yolov3 data 정보가 담긴 .cfg 파일을 block으로 변환
2. def load_classes : pth 파일로부터 class를 load
3. def create_train_data_loader : train dataloader
4. def create_validation_data_loader : validation dataloader
+ +model save/load 관련 기능을 추가로 구현 예정
### layers.py
yolov3에 들어가는 layer들을 정의하는 모듈
1. class EmptyLayer : empty layer
2. class YOLOV3Layer : Yolo layer
3. def block_to_layer : layer 정보가 담긴 block을 layer로 변환
### loss.py
loss를 계산하는 모듈
1. def compute_yolo_loss : loss를 계산
2. def build_targets : compute_yolo_loss에 필요한 기능 제공 
### logger.py
log 기능을 제공
1. class Logger : Logger class
### evaluate.py
evaluate 기능을 제공
1. def yolo_evaluate : yolo model evaluate
2. def print_eval_stats : yolo_evaluate에 필요한 기능 제공(1에 끼워넣어도 될듯)

## layer 3
### yoloutils.py
yolov3에 필요한 편의적인 기능들을 제공하는 모듈
### utils.py
전반적으로 편리한 기능들을 제공
