[![Build Status](https://travis-ci.org/keon/3-min-pytorch.svg?branch=master)](https://travis-ci.org/keon/3-min-pytorch)

# 펭귄브로의 3분 딥러닝, 파이토치맛
[PyTorch 코드로 맛보는 CNN, GAN, RNN, DQN, Autoencoder, ResNet, Seq2Seq, Adversarial Attack](https://book.naver.com/bookdb/book_detail.nhn?bid=15559613)

> 저자: [김건우](https://github.com/keon), [염상준](https://github.com/ysangj)

파이토치 코드로 맛보는 딥러닝 핵심 개념! 

이 책은 파이토치로 인공지능을 구현하는 방법을 알려줍니다. 인공지능 입문자를 위한 기초 지식과 최신 인공지능 구현 방법인 인공신경망 기술을 사례를 통해 알아봅니다. 지도학습 방식의 ANN, DNN, CNN, RNN을 비롯해, 비지도학습 방식의 AE와 GAN 그리고 강화학습 DQN을 직접 구현합니다. 딥러닝의 약점을 이용해서 해킹하는 방법을 배우는 등 각 장에서 소개한 신경망으로 재미있는 응용 예제를 다룹니다.

<a href="http://www.yes24.com/Product/Goods/80218706">
<p align="center">
<img src="http://www.hanbit.co.kr/data/books/B7193109877_l.jpg" alt="3분 딥러닝 파이토치맛" title="3분 딥러닝 파이토치맛" width="350"/>
</p>
</a>

- [네이버책](https://book.naver.com/bookdb/book_detail.nhn?bid=15559613), 
[yes24](http://www.yes24.com/Product/Goods/80218706),
[교보문고](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791162242278&fbclid=IwAR1RuBmj9OKXmqi291yddZ53yVkPl3bkOqJKgGbu0tKDlq8MijjN7xiUAvs#N),
[알라딘](https://www.aladin.co.kr/shop/wproduct.aspx?ISBN=K002636987),
[영풍문고](http://www.ypbooks.co.kr/book.yp?bookcd=100983591)
[인터파크도서](http://book.interpark.com/product/BookDisplay.do?_method=detail&sc.prdNo=318434586),
[반디앤루니스](http://www.bandinlunis.com/front/product/detailProduct.do?prodId=4284510&compId=101) 등에서 만나볼 수 있습니다. 


## 요구사항

아래 파이토치와 파이썬 버전을 지원합니다.

* PyTorch 1.0 이상
* Python 3.6.1 이상


## 구성

이 책은 딥러닝과 파이토치를 처음 접하는 사람이 쉽게 이론을 익히고 구현할 수 있도록 구성돼 있습니다. 딥러닝은 언어부터 이미지까지 넓은 분야에 사용되고 있어서 응용하는 분야에 따라 그 형태가 다양합니다. 따라서 최대한 다양한 학습 방식과 딥러닝 모델을 구현할 수 있도록 예제를 준비했습니다.

### [1장. 딥러닝과 파이토치](./01-딥러닝과_파이토치)

딥러닝의 기본 지식을 쌓고, 여러 기계학습 방식에 대해 배웁니다. 파이토치가 무엇이고, 왜 필요한지와, 텐서플로와 케라스 같은 라이브러리와 무엇이 다른지에 대해 알아봅니다.

### [2장. 파이토치 시작하기](./02-파이토치_시작하기)

파이토치 환경 설정과 사용법을 익혀봅니다. 파이토치 외에도 책을 진행하면서 필요한 주변 도구를 설치합니다.

### [3장. 파이토치로 구현하는 ANN](./03-파이토치로_구현하는_ANN)

파이토치를 이용하여 가장 기본적인 인공 신경망을 구현하고 모델을 저장, 재사용하는 방법까지 배웁니다.

### [4장. 패션 아이템을 구분하는 DNN](./04-패션_아이템을_구분하는_DNN)

앞서 배운 인공 신경망을 이용하여 Fashion MNIST 데이터셋 안의 패션 아이템을 구분해봅니다.

### [5장. 이미지 처리능력이 탁월한 CNN](./05-이미지_처리능력이_탁월한_CNN)

영상 인식에 탁월한 성능을 자랑하는 CNN에 대하여 알아봅니다. 여기에 그치지 않고 CNN을 더 쌓아 올려 성능을 올린 ResNet에 대해 알아보고 구현합니다.

### [6장. 사람의 지도 없이 학습하는 오토인코더](./06-사람의_지도_없이_학습하는_오토인코더)

정답이 없는 상태에서 특징을 추출하는 비지도학습에 대해 알아보고 대표적인 비지도학습 모델인 오토인코더를 이해하고 구현하는 방법을 익힙니다.

### [7장. 순차적인 데이터를 처리하는 RNN](./07-순차적인_데이터를_처리하는_RNN)

문자열, 음성, 시계열 데이터에 높은 성능을 보이는 RNN을 활용하여 영화 리뷰 감정 분석을 해보고 간단한 기계 번역기를 만들어봅니다.

### [8장. 딥러닝을 해킹하는 적대적 공격](./08-딥러닝을_해킹하는_적대적_공격)

딥러닝 모델을 의도적으로 헷갈리게 하는 적대적 예제에 대해 알아보고 적대적 예제를 생성하는 방법인 적대적 공격(adversarial attack)을 알아봅니다.

### [9장. 경쟁하며 학습하는 GAN](./09-경쟁하며_학습하는_GAN)

두 모델의 경쟁을 통해 최적화하는 특이한 학습 구조를 가진 GAN에 대해 알아봅니다. GAN은 데이터셋에 존재하지 않는 새로운 이미지를 생성할 수 있습니다. 예제로 Fashion MNIST 데이터셋을 학습하여 새로운 패션 아이템을 만듭니다.

### [10장. 주어진 환경과 상호작용하며 성장하는 DQN](./10-주어진_환경과_상호작용하며_성장하는_DQN)

간단한 게임 환경에서 스스로 성장하는 DQN에 대해 알아보고 간단한 게임을 마스터하는 인공지능을 구현해봅니다.


## 참여하기

**`중요!`** 모든 코드는 주피터 노트북 파일인 `.ipynb`로 쓰여져야 합니다.

주피터 노트북으로 작성 후 `compile_notebook.py`를 실행시키면 주석과 코드 모두 파이썬 파일로 예쁘게 변환됩니다.

일반 파이썬 포멧으로 쓰여진 `.py` 파일은 변환과정에서 삭제될 수 있으니 주의바랍니다.


## 참고

* [홍콩과기대 김성훈 교수님의 모두를 위한 머신러닝/딥러닝 강의](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
* [앤드류응 교수님의 Neural Network and Deep Learning 강의](https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
* [뉴욕대 조경현 교수님의 Brief Introduction to Machine Learning without Deep Learning 수업노트](https://github.com/nyu-dl/Intro_to_ML_Lecture_Note)
* [뉴욕대 조경현 교수님의 Natural Language Understanding with Distributed Representation 수업노트](https://github.com/nyu-dl/NLP_DL_Lecture_Note)
* [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - MIT License
* [jcjohnson/pytorch-examples](https://github.com/jcjohnson/pytorch-examples) - MIT License
* [Deep Learning in a Nutshell: Core Concepts](https://devblogs.nvidia.com/deep-learning-nutshell-core-concepts/)
* [이찬우님의 딥러닝 비디오](https://www.youtube.com/channel/UCRyIQSBvSybbaNY_JCyg_vA/videos)
* [CS231n: Convolutional Neural Networks for Visual Recognition 강의노트의 한글 번역 버전](http://aikorea.org/cs231n)
