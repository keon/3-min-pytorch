is_ready = True
installed_packages = []
uninstalled_packages = []
try:
    import torch
    installed_packages.append("파이토치 버전:%s" % torch.__version__)
except:
    is_ready = False
    uninstalled_packages.append("파이토치")
try:
    import torchvision
    installed_packages.append("토치비젼 버전:%s" % torchvision.__version__)
except:
    is_ready = False
    uninstalled_packages.append("토치비전")
try:
    import torchtext
    installed_packages.append("토치텍스트 버전:%s" % torchtext.__version__)
except:
    is_ready = False
    uninstalled_packages.append("토치텍스트")
try: 
    import numpy
    installed_packages.append("넘파이 버전:%s" % numpy.__version__)
except:
    is_ready = False 
    uninstalled_packages.append("넘파이")
try:
    import matplotlib
    installed_packages.append("맷플랏립 버전:%s" % matplotlib.__version__)
except:
    is_ready = False
    uninstalled_packages.append("맷플랏립")
try:
    import sklearn
    installed_packages.append("사이킷런 버전:%s" % sklearn.__version__)
except:
    is_ready = False
    uninstalled_packages.append("사이킷런")

if is_ready:
    print("축하합니다! 3분 딥러닝 파이토치맛 예제 실행을 위한 환경설정이 끝났습니다.")
    print("설치된 라이브러리 정보:")
    for pkg in installed_packages:
        print(" * " + pkg)
else:
    print("미설치된 라이브러리가 있습니다.")
    print("설치된 라이브러리 정보:")
    for pkg in installed_packages:
        print(" * " + pkg)
    print("미설치된 라이브러리 정보:")
    for pkg in uninstalled_packages:
        print(" * " + pkg)

