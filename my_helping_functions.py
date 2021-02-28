import os

from my_constants import *

def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.readlines()

def read_files(files):
    tmp=[]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                tmp.append(line)

    return tmp

def read_files_into_separate_lists(files):
    tmp=[]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            tmp.append(f.readlines())

    return tmp

def write_file(file, lines):
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return None

def write_files(dict):
    # keys are file names
    # values are list of lines to write
    for file in dict.keys():
        with open(file, 'w', encoding='utf-8') as f:
            f.writelines(dict[file])
    return None

def parallelize_list(l, n_cores=os.cpu_count()):
    parallelizable_data=[]
    if l:
        step_size=int(len(l)/n_cores)
        for i in range(0, len(l), step_size):
            parallelizable_data.append(l[i:i+step_size]) if i+step_size < len(l) else parallelizable_data.append(l[i:])

    return parallelizable_data

def serialize_list(l):
    # Note: it serializes only one level.
    # for example, 
    # l=[[1, 2], [3, 4]]
    # will be l=[1, 2, 3, 4]
    serial_l=[]
    for sublist in l:
        for line in sublist:
            serial_l.append(line)

    return serial_l

def convert_into_dictionary(keys, values):
    dict={}
    for k, v in zip(keys, values):
        dict[k]=v

    return dict
def get_extension(file, char_before_extension=DOT):
    return file[file.rfind(char_before_extension)+1:]

def get_main_data_files(dir, src, tgt):
    allowed_files=(TRAIN, VALID, TEST)
    allowed_extensions=(src, tgt)
    files=[]
    for file in os.listdir(dir):
        tmp=file.split(DOT)
        if tmp[0] in allowed_files and tmp[1] in allowed_extensions:
            files.append(dir+file)
    return files

def get_splitted_data_directories(dir, src, tgt):
    allowed_files=(TRAIN, VALID, TEST)
    allowed_extensions=(src, tgt)
    files=[]
    for file in os.listdir(dir):
        tmp=file.split('_')
        if tmp[0] in allowed_files and tmp[1] in allowed_extensions:
            files.append(dir+file)
    return files

def get_segmented_files_in(dir):
    tmp=[]
    for file in os.listdir(dir):
        if '-segmented' in file:
            tmp.append(dir+file)

    return tmp

def get_postsegmented_files_in(dir):
    tmp=[]
    for file in os.listdir(dir):
        if '-postsegmented' in file:
            tmp.append(dir+file)

    return tmp

def convert_name_into_segmented_name(file):
    lang=get_extension(file)
    return file+'-segmented.'+lang

def convert_name_into_postsegmented_name(file):
    lang=get_extension(file)
    if '-segmented' in file:
        file=file[:file.rfind('-segmented')]
    return file+'-postsegmented.'+lang

def convert_name_into_bped_name(file):
    lang=get_extension(file)
    file=file[:file.rfind('-')+1]
    return file+'bpe.'+lang





def wait_till_finish_writing(file, reference_file=None):
    from subprocess import Popen, PIPE
    import shlex
    from time import sleep
    if not reference_file:
        reference_file=file[:file.rfind('-')]
    reference_num_lines=None
    while not reference_num_lines:
        reference_num_lines=Popen(shlex.split('wc -l '+reference_file), stdout=PIPE, stderr=PIPE).communicate()[0].decode('ascii')
    reference_num_lines=int(reference_num_lines.split()[0])
    num_lines=0
    while num_lines!=reference_num_lines:
        cmd_out=None
        while not cmd_out:
            cmd_out=Popen(shlex.split('wc -l '+file), stdout=PIPE, stderr=PIPE).communicate()[0].decode('ascii')
        if 'No such file or directory' not in cmd_out:
            num_lines=int(cmd_out.split()[0])
        sleep(1)

    return True
