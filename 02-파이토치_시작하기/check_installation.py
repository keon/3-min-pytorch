from importlib import import_module

required_packages = [
    ("torch", "파이토치"),
    ("torchvision", "토치비전"),
    ("torchtext", "토치텍스트"),
    ("numpy", "넘파이"), 
    ("matplotlib", "맷플롯립"),
    ("sklearn", "사이킷런"),
]
installed_packages = []
uninstalled_packages = []

def run():
    is_ready = True
    for package_name, korean_name in required_packages:
        printed_name = "%s(%s)" % (korean_name, package_name)
        try:
            imported_package = import_module(package_name)
            installed_packages.append("%s 버전:%s" % (printed_name, imported_package.__version__))
        except:
            is_ready = False
            uninstalled_packages.append(printed_name)

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
    return is_ready 


if __name__ == "__main__":
    run()