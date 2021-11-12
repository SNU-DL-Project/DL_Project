# 주의사항
### data/custom에 images, labels 폴더 넣기 (ex : data/custom/images/P0000.jpg)  
### pytorch 버전에 따라 오류가 생길 가능성 존재.  
### 사용중인 GPU 용량에 따라 오류가 생길 존재. cfg file의 # Training 밑의 batch를 2 or 4가 아닌 1로 수정.

# 사용법
main.py 실행  
main.py에 적힌 주석 참고  
아직 완벽히 구현되지 않아, 구체적인 설명은 첨부하지 않았습니다. 

# Error Report
(*오류가 생기면 여기에 작성시 반영하겠습니다.)

# v0.1.0 (2021.11.10.)  
### 주요 변경사항
YOLOv3 구현

# v0.1.0 (2021.11.12.)  
### 주요 변경사항
SRResnet 구현 (SRGAN은 Pruning이 힘들 듯)
### 기타 변경사항
+ Train Class training path 변수화, loggers 분리  
+ Train Class 함수 train_on_step(FP, BP 통합), train_init 구현(run 함수 간단화)  
+ SRResnet optimizer 수정 (YOLOv3와 같은 형태)
