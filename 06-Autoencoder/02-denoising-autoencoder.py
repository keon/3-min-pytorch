#!/usr/bin/env python
# coding: utf-8

# # 6.2 오토인코더로 망가진 이미지 복원하기
# 잡음제거 오토인코더(Denoising Autoencoder)는 2008년 몬트리올 대학에서 발표한 논문
# ["Extracting and Composing Robust Features with Denoising AutoEncoder"](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
# 에서 처음 제안되었습니다.
# 앞서 오토인코더는 일종의 "압축"을 한다고 했습니다.
# 그리고 압축은 데이터의 특성에 중요도로 우선순위를 매기고
# 낮은 우선순위의 데이터를 버린다는 뜻이기도 합니다.
# 잡음제거 오토인코더의 아이디어는
# 중요한 특징을 추출하는 오토인코더의 특성을 이용하여 비교적
# "덜 중요한 데이터"인 잡음을 버려 원래의 데이터를 복원한다는 것 입니다.
# 원래 배웠던 오토인코더와 큰 차이점은 없으며,
# 학습을 할때 입력에 잡음을 더하는 방식으로 복원 능력을 강화한 것이 핵심입니다.



