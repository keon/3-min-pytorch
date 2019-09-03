import sys
import os
from importlib import import_module


# setup path
chapter_name = "04-패션_아이템을_구분하는_DNN"
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".." , chapter_name)
sys.path.append(dir_path)



def test_chapter_exmaples():
    mydir_tmp = os.path.join(dir_path) # add the testA folder name
    mydir_new = os.chdir(mydir_tmp) # change the current working directory
    mydir = os.getcwd()  # set the main directory again, now it calls testA

    chapter_examples = [
        "fashion_mnist",
        "neural_network",
        "overfitting_and_regularization",
    ]

    for example in chapter_examples:
        imported_package = import_module(example)

if __name__ == "__main__":
    test_chapter_exmaples()