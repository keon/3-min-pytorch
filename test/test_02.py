import sys 
import os 


# setup path
chapter_name = "02-파이토치_시작하기"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".." , chapter_name)
sys.path.append(dir_path)

import check_installation

def test_check_installation():
    assert check_installation.run() == True