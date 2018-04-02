import os
import numpy as np
import pandas as pd
import random
import datetime
import fnmatch
import psutil

np.set_printoptions(suppress=True)


def sublistfinder(mylist, pattern):
    """
    Find the location of sub-list in a python list, no repetitions in mylist or pattern
    :param mylist: longer list
    :param pattern: shorter list
    :return: list of locations of elements in mylist ordered as in pattern
    """
    return [mylist.index(x) for x in pattern if x in mylist]


def dir_create(input_dir):
    """
    Create dir if it doesnt exist
    :param: directory to be created
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        # print('Created path: ' + input_dir)
    else:
        # print('Path already exists: ' + input_dir)
        pass


def read_csv_as_array(csv_file):
    """
    Read csv file as numpy array
    """
    csv_data = np.genfromtxt(csv_file, delimiter=',', names=True, dtype=None)
    names = [csv_data.dtype.names[i] for i in range(0, len(csv_data.dtype.names))]
    return names, csv_data


def read_text_by_line(infile):
    """
    Read text file line by line and output a list of text lines
    :param infile:
    :return:
    """
    with open(infile) as f:
        content = f.readlines()
    return [x.strip() for x in content]


def file_rename_check(filename):
    """
    Change the file name if already exists by incrementing trailing number
    :param filename
    :return filename
    """
    if not os.path.isfile(filename):
        dir_create(os.path.dirname(filename))

    # if file exists then rename the input filename
    counter = 1
    while os.path.isfile(filename):
        filebasename = os.path.basename(filename)
        filedirname = os.path.dirname(filename)
        components = filebasename.split('.')
        if len(components) < 2:
            filename = filedirname + os.path.sep + filebasename + '_' + str(counter)
        else:
            filename = filedirname + os.path.sep + ''.join(components[0:-1]) + \
                       '_(' + str(counter) + ').' + components[-1]
        counter = counter + 1
    return filename


def file_remove_check(filename):
    """
    Remove a file silently; if not able to remove, change the filename and move on
    :param filename
    :return filename
    """

    # if file does not exist, try to create dir
    if not os.path.isfile(filename):
        dir_create(os.path.dirname(filename))

    # if file exists then try to delete or
    # get a filename that does not exist at that location
    counter = 1
    while os.path.isfile(filename):
        # print('File exists: ' + filename)
        # print('Deleting file...')
        try:
            os.remove(filename)
        except OSError:
            filebasename = os.path.basename(filename)
            filedirname = os.path.dirname(filename)
            components = filebasename.split('.')
            if len(components) < 2:
                filename = filedirname + os.path.sep + filebasename + '_' + str(counter)
            else:
                filename = filedirname + os.path.sep + ''.join(components[0:-1]) + \
                           '_(' + str(counter) + ').' + components[-1]
            # print('Unable to delete, using: ' + filename)
            counter = counter + 1
    return filename


def write_list_to_file(filename, input_list):
    """
    Function to write list to file
    with each list item as one line
    """

    # create dir path if it does not exist
    dirname = os.path.dirname(filename)
    dir_create(dirname)

    # if file exists, delete it
    filename = file_remove_check(filename)

    # write to file
    with open(filename, 'w') as fileptr:
        for line in input_list:
            fileptr.write('%s\n' % line)


# write slurm script for the given parameters
def write_slurm_script(script_filename, job_name='pyscript', time_in_mins=60,
                       cpus=1, ntasks=1, mem=2000,
                       array=False, iterations=1, **kwargs):
    script_dict = {
        'bash': '#!/bin/bash',
        'job-name': '#SBATCH --job-name=' + job_name,
        'time': '#SBATCH --time=' + str(datetime.timedelta(minutes=time_in_mins)),
        'cpus': '#SBATCH --cpus-per-task=' + str(cpus),
        'ntasks': '#SBATCH --ntasks=' + str(ntasks),
        'mem': '#SBATCH --mem=' + str(mem),
        'partition': '#SBATCH --partition=all',
        'array_def': '#SBATCH --array=1-' + str(iterations),
        'array_out': '#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/iter_%A_%a.out',
        'out': '#SBATCH --output=/scratch/rm885/support/out/slurm-jobs/slurm_%j.out',
        'date': 'date',
        'array_echo': 'echo "Job ID is"$SLURM_ARRAY_TASK_ID'
    }
    if array:
        script_list = [
            script_dict['bash'],
            script_dict['job-name'],
            script_dict['time'],
            script_dict['cpus'],
            script_dict['mem'],
            script_dict['partition'],
            script_dict['array_def'],
            script_dict['array_out'],
            script_dict['date'],
        ]
        for key, value in kwargs.items():
            script_list.append(value)
        script_list.append(script_dict['date'])
    else:
        script_list = [
            script_dict['bash'],
            script_dict['job-name'],
            script_dict['time'],
            script_dict['cpus'],
            script_dict['mem'],
            script_dict['partition'],
            script_dict['out'],
            script_dict['date'],
        ]
        for key, value in kwargs.items():
            script_list.append(value)
        script_list.append(script_dict['date'])

    write_list_to_file(script_filename, script_list)


# write numpy array to file
def write_numpy_array_to_file(filename, np_array):
    colnames = list(np_array.dtype.names)
    np.savetxt(filename, np_array, delimiter=",", fmt='%3.9f', header=','.join(colnames), comments='')


def read_from_csv(csv_file):
    """
    Read csv as list of lists with header.
    Each row is a list and a sample point.
    :param: csv file
    :returns: dictionary of data
    """
    dataframe = pd.read_csv(csv_file, sep=',')

    # convert col names to list of strings
    names = dataframe.columns.values.tolist()

    # convert pixel samples to list
    features = dataframe.values.tolist()
    return {
        'feature': features,
        'name': names,
    }


def find_all(pattern, path):
    """Find all the names that match pattern"""

    result = []
    # search for a given pattern in a folder path
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, '*' + pattern + '*'):
                result.append(os.path.join(root, name))
    return result  # list


def print_memory_usage():
    """
    Function to print memory usage of the python process
    :return print to console
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss

    if 1024 <= mem < 1024 * 1024:
        div = 1024.0
        suff = ' KB'
    elif 1024 * 1024 <= mem < 1024 * 1024 * 1024:
        div = 1024.0 * 1024.0
        suff = ' MB'
    elif 1024 * 1024 * 1024 <= mem < 1024 * 1024 * 1024 * 1024:
        div = 1024.0 * 1024.0 * 1024.0
        suff = ' GB'
    elif mem >= 1024 * 1024 * 1024 * 1024:
        div = 1024.0 * 1024.0 * 1024.0 * 1024.0
        suff = ' TB'
    else:
        div = 1.0
        suff = ' BYTES'

    print_str = 'MEMORY USAGE: {:{w}.{p}f}'.format(process.memory_info().rss / div, w=5, p=2) + suff

    print('')
    print('*******************************************')
    print(print_str)
    print('*******************************************')
    print('')


def list_size(query_list):
    """Find size of a list object"""

    if isinstance(query_list, list):
        return len(query_list)
    else:
        return 1
