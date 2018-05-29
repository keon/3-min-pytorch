# 3분 딥러닝 파이토치맛

> 저자: [김건우](https://github.com/keon), [염상준](https://github.com/ysangj)

## 요구사항

* PyTorch 1.0
* Python >= 3.6.1


## 목차

1. [딥러닝과 파이토치](01-Deep-Learning-And-PyTorch) - 딥러닝의 기본 지식을 쌓고 파이토치의 장단점에 대해 알아봅니다.
  * [개념] 신경망의 원리
  * [개념] 딥러닝과 신경망
  * [개념] 왜 파이토치인가?
2. [파이토치 시작하기](02-Getting-Started-With-PyTorch) - 파이토치 환경설정과 사용법을 익혀봅니다
  * [프로젝트 1] 파이토치 설치 & 환경구성
  * [프로젝트 2] 파이토치 예제 내려받고 실행해보기
  * [프로젝트 3] 토치비전과 토치텍스트로 데이터셋 불러오기
3. [파이토치로 구현하는 신경망](03-Coding-Neural-Networks-In-PyTorch) - 파이토치를 이용하여 가장 기본적인 신경망을 만들어봅니다.
  * [개념] 텐서와 Autograd
  * [Hello World] 신경망 모델 구현하기
  * [Hello World] 모델 저장, 재사용
4. [딥러닝으로 패션 아이템 구분하기](04-Neural-Network-For-Fashion) - Fashion MNIST 데이터셋과 앞서 배운 인공신경망을 이용하여 패션아이템을 구분해봅니다.
  * [개념] [Fashion MNIST 데이터셋 설명](04-Neural-Network-For-Fashion/01-fasion-mnist.ipynb)
  * [프로젝트 1] [Fashion MNIST 학습하기](04-Neural-Network-For-Fashion/02-neural-network.ipynb)
  * [팁] 성능 측정법 알아보기 (Train/Validation/Test)
  * [프로젝트 2] [오버피팅과 정규화](04-Neural-Network-For-Fashion/02-overfitting-and-regularization.ipynb)
  * 더 보기
5. [이미지 인식능력이 탁월한 CNN](05-CNN-For-Image-Classification)
  * [개념] CNN 기초
  * [프로젝트 1] [CNN 모델 구현하기](05-CNN-For-Image-Classification/01-cnn.ipynb)
  * [프로젝트 2] [컬러 데이터셋에 적용하기](05-CNN-For-Image-Classification/02-cifar-cnn.ipynb)
  * [팁] [토치비전으로 복잡한 모델 사용하기](05-CNN-For-Image-Classification/03-torcivision-models.ipynb)
  * 더 보기
6. [신경망 깊게 쌓아보기](06-Getting-Deeper) - CNN의 발전사와 함께 발전된 형태의 모델들을 알아봅니다.  
  * [개념] 복잡한 CNN모델들
  * [개념 or 프로젝트] Alexnet
  * [개념 or 프로젝트] Residual Networks (ResNet)
  * [개념 or 프로젝트] Inception
  * [프로젝트] High Level API 사용법 익히기
  * 더 보기
7. [사람의 지도 없이 학습하는 Autoencoder](07-Autoencoder) - 레이블이 없는 상태서 특징추출을 하는 오토인코더에 대해 배워봅니다.
  * [개념] 오토인코더 기초
  * [프로젝트 1] 오토인코더로 이미지의 특징을 압축해보기
  * [프로젝트 2] Latent 공간 탐험하기
  * 더 보기
8. [순차적인 데이터를 처리하는 RNN](08-RNN-For-Sequential-Data) - RNN을 활용하여 영화 리뷰 감정 분석과 기계번역을 해봅니다
  * [개념] RNN 기초
  * [프로젝트 1] 영화 리뷰 감정 분석
  * [프로젝트 2] Seq2Seq 기계 번역
  * 더 보기
9. [딥러닝 해킹하기](09-Hacking-Deep-Learning) - Adversarial Attack
  * [개념] Adversarial Attack 이란?
  * [프로젝트 1] FGSM 공격
  * [프로젝트 2] 타겟을 정해 공격하기
  * 더 보기
10. [경쟁을 통해 성장하는 GAN](10-Generative-Adversarial-Networks) - GAN을 이용하여 새로운 패션 아이템을 만들어봅니다.
  * [개념] GAN 기초
  * [프로젝트 1] GAN으로 새로운 패션아이템 생성하기
  * [프로젝트 2] Conditional GAN으로 생성 컨트롤하기
  * 더 보기
11. [주어진 환경과 상호작용을 통해 성장하는 DQN](11-DQN-Learns-From-Environment) - 간단한 게임환경 안에서 스스로 성장하는 DQN 에이전트를 만들어봅니다.
  * [개념] 강화학습과 DQN기초
  * [팁] OpenAI Gym
  * [프로젝트 1] 카트폴 게임 마스터하기
  * 더 보기
12. [A2C로 간단한 자율주행 해보기](12-A2C-Self-Driving) - A2C알고리즘을 활용하여 간단한 자율주행차를 만들어봅니다.
  * [개념] 자율주행차란?
  * [팁] 자율주행 시뮬레이터 소개
  * [팁] 설치와 환경설정
  * [프로젝트 1] 딥러닝으로 자동차 조종하기
  * 더 보기

## 참여하기

**`중요!`** 모든 코드는 주피터 노트북 파일인 `.ipynb`로 쓰여져야 합니다.

주피터 노트북으로 작성 후 `compile_notebook.py`를 실행시키면 주석과 코드 모두 파이썬 파일로 예쁘게 변환됩니다.

일반 파이썬 포멧으로 쓰여진 `.py` 파일은 변환과정에서 삭제될 수 있으니 주의바랍니다.


## 참고

[홍콩과기대 김성훈 교수님의 모두를 위한 머신러닝/딥러닝 강의](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
