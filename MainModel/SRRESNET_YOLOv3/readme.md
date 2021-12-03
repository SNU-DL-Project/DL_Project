# 주의사항
### 구글드라이브의 data 폴더를 그대로 넣기
### pytorch 버전에 따라 오류가 생길 가능성 존재.  
### 사용중인 GPU 용량에 따라 오류가 생길 존재. cfg file의 # Training 밑의 batch를 2 or 4가 아닌 1로 수정.

# 사용법
v0.x.x 폴더 들어간 후, 구글드라이브에 있는 data 폴더를 넣기  
main.py 실행 및 main.py에 적힌 주석 참고  
아직 완벽히 구현되지 않아, 구체적인 설명은 첨부하지 않았습니다. 

# Error Report
(*오류가 생기면 여기에 작성시 반영하겠습니다.)

# v0.1.0 (2021.11.10.)  
### 주요 변경사항
##### YOLOv3 구현

# v0.2.0 (2021.11.12.)  
### 주요 변경사항
##### SRResnet 구현 (SRGAN은 Pruning이 힘들 듯)
### 기타 변경사항
+ Train Class training path 변수화, loggers 분리  
+ Train Class 함수 train_on_step(FP, BP 통합), train_init 구현(run 함수 간단화)  
+ SRResnet optimizer 수정 (YOLOv3와 같은 형태)

# v0.2.1 (2021.11.12.)  
### 기타 변경사항
+ labels 관련 문제 해결

# v0.3.0 (2021.11.21.)  
### 주요 변경사항
##### darknet.py, srmodels.py를 models.py로 통합  
##### SRRES upsample 2->4로 변경  
##### 예비실험 결과 바탕으로 config file, loss 비율 등 hyperparameter 조절
### 기타 변경사항
+ downsample할 때 Gaussian Blur 처리  
+ train 후 모델 저장 기능 추가  
+ augmentation normal -> strong 변경 (더 강한 augmentation)  
+ input size 관련 오류 수정  

# v0.3.1 (2021.11.25.)
### 기타 변경사항
+ yolov3, srres cfg파일 수정  
+ class별 AP evaluate 기능 추가  
+ labels 수정

# v0.4.0 (2021.11.26.)
### 주요 변경사항  
##### SRRES+YOLOv3 모델 구현  
### 기타 변경사항  
+ loss파일 loss output 관련 수정  
+ 예비실험 결과 반영(config, blur)  

# v0.5.0 (2021.12.03.)
### 주요 변경사항  
##### test_SRutils.py 구현완료 및 추가  
##### test_Yolov3utils.py 구현완료 및 추가  
##### test.py 구현미완료(연결작업 필요) 및 추가  
