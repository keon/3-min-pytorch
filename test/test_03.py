import sys
import os
from importlib import import_module


# setup path
chapter_name = "03-파이토치로_구현하는_ANN"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".." , chapter_name)
sys.path.append(dir_path)

def test_chapter_exmaples():
    mydir_tmp = os.path.join(dir_path) # add the testA folder name
    mydir_new = os.chdir(mydir_tmp) # change the current working directory
    mydir = os.getcwd() # set the main directory again, now it calls testA

    chapter_examples = [
        "tensor_basic",
        "tensor_operation",
        "autograd_basic",
        "image_recovery",
        "basic_neural_network",
    ]

    for example in chapter_examples:
        imported_package = import_module(example)

if __name__ == "__main__":
    test_chapter_exmaples()