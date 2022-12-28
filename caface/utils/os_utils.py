import os
import shutil
import re

def get_all_files(root, extension_list=['.csv']):

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files


def copy_project_files(code_dir, save_path):

    print('copying files from {}'.format(code_dir))
    print('copying files to {}'.format(save_path))
    py_files = get_all_files(code_dir, extension_list=['.py'])
    os.makedirs(save_path, exist_ok=True)
    for py_file in py_files:
        os.makedirs(os.path.dirname(py_file.replace(code_dir, save_path)), exist_ok=True)
        shutil.copy(py_file, py_file.replace(code_dir, save_path))

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
