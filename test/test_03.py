import sys
import os


# setup path
chapter_name = "03-파이토치로_구현하는_신경망_ANN"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".." , chapter_name)
sys.path.append(dir_path)

import check_installation

def test_check_installation():
    assert check_installation.run() == True