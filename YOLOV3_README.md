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
> main.py : API, 확인용  

### (layer 1)
> darknet.py : weight 로드, darknet을 모델링한다.  
> train.py : data를 load하고 train 한다. 기록(loss, weight)을 print, 저장  
> test.py : test  

### (layer 2)
> dataload.py : 데이터 로드  
> layers.py : yolo layer 정의, configuration 파일을 layer로 변환  
> config.py : configuration 파일을 리스트로 읽어오기  
> loss.py : loss를 계산  
> logger.py : 파이토치 log를 print, 저장하는 기능  
> (!)yolov3_evaluate.py : evaluation을 진행. test, train 등에 사용 (모듈 이름 수정 필요(너무 김))  
> (!)optm.py : optimizer 정의 (모듈 이름 수정 필요)  

### (layer 3)
> yolov3_utils.py : yolov3에 필요한 기타 기능들 (IoU 계산, 좌표변환 등)  
> common_utils.py : 일반적으로 필요로하는 기능들 (to_CPU 등)  

## layer 0
### main.py
main API
## layer 1
### darknet.py
yolov3의 네트워크 모듈
1. class Darknet  
+ forward 
+ parameter save 기능 (구현 예정)
### train.py
train을 진행하는 모듈
1. det run
+ parameter 값에 따라 train 진행
### test.py
test를 진행하는 모듈
1. det run
+ test dataset에 대해 test 진행
## layer 2
### dataload.py
train, val 데이터를 입력받는 모듈
### layers.py
yolov3에 들어가는 layer들을 정의하는 모듈
1. class EmptyLayer  
+ empty layer
2. class YOLOV3Layer  
+ Yolo layer
3. def block_to_layer  
+ layer 정보가 담긴 block을 layer로 변환
## layer 3
