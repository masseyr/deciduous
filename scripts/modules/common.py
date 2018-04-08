import os
import numpy as np
import pandas as pd
import datetime
import fnmatch
import psutil
import ftplib

np.set_printoptions(suppress=True)


__all__ = ['Sublist',
           'Handler',
           'FTPHandler',
           'Opt']


class Sublist(list):
    """
    Class to handle list operations
    """
    @staticmethod
    def list_size(query_list):
        """Find size of a list object"""

        if isinstance(query_list, list):
            return len(query_list)
        else:
            return 1

    def sublistfinder(self, pattern):
        """
        Find the location of sub-list in a python list, no repetitions in mylist or pattern
        :param pattern: shorter list
        :return: list of locations of elements in mylist ordered as in pattern
        """
        return [self.index(x) for x in pattern if x in self]


class Handler(object):
    """
    Class to handle file and folder operations
    """
    def __init__(self,
                 filename=None,
                 basename=None,
                 dirname=None):

        self.filename = filename

        try:
            self.basename = os.path.basename(filename)
        except:
            self.basename = basename

        try:
            self.dirname = os.path.dirname(filename)
        except:
            self.dirname = dirname

        self.sep = os.path.sep

    def __repr__(self):
        if self.filename is not None:
            return '<Handler for {}>'.format(self.filename)
        elif self.dirname is not None:
            return '<Handler for {}>'.format(self.dirname)
        elif self.basename is not None:
            return '<Handler for {}>'.format(self.basename)
        else:
            return '<Handler {s}____{s}>'.format(s=self.sep)

    def dir_create(self):
        """
        Create dir if it doesnt exist
        :param: directory to be created
        """
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
            # print('Created path: ' + input_dir)
        else:
            print('Path already exists: ' + self.dirname)

    def file_rename_check(self):
        """
        Change the file name if already exists by incrementing trailing number
        :return filename
        """
        if not os.path.isfile(self.filename):
            self.dir_create()

        # if file exists then rename the input filename
        counter = 1
        while os.path.isfile(self.filename):
            components = self.basename.split('.')
            if len(components) < 2:
                self.filename = self.dirname + os.path.sep + self.basename + '_' + str(counter)
            else:
                self.filename = self.dirname + os.path.sep + ''.join(components[0:-1]) + \
                           '_(' + str(counter) + ').' + components[-1]
            counter = counter + 1
        return self.filename

    def file_remove_check(self):
        """
        Remove a file silently; if not able to remove, change the filename and move on
        :return filename
        """

        # if file does not exist, try to create dir
        if not os.path.isfile(self.filename):
            self.dir_create()

        # if file exists then try to delete or
        # get a filename that does not exist at that location
        counter = 1
        while os.path.isfile(self.filename):
            # print('File exists: ' + filename)
            # print('Deleting file...')
            try:
                os.remove(self.filename)
            except OSError:
                components = self.basename.split('.')
                if len(components) < 2:
                    self.filename = self.dirname + os.path.sep + self.basename + '_' + str(counter)
                else:
                    self.filename = self.dirname + os.path.sep + ''.join(components[0:-1]) + \
                               '_(' + str(counter) + ').' + components[-1]
                # print('Unable to delete, using: ' + filename)
                counter = counter + 1
        return self.filename

    def file_exists(self):
        """
        Check file existence
        :return: Bool
        """
        return os.path.isfile(self.filename)

    def dir_exists(self):
        """
        Check folder existence
        :return: Bool
        """
        return os.path.isdir(self.dirname)

    def file_delete(self):
        """
        Delete a file
        """
        os.remove(self.filename)

    def read_csv_as_array(self):
        """
        Read csv file as numpy array
        """
        csv_data = np.genfromtxt(self.filename, delimiter=',', names=True, dtype=None)
        names = [csv_data.dtype.names[i] for i in range(0, len(csv_data.dtype.names))]
        return names, csv_data

    def read_text_by_line(self):
        """
        Read text file line by line and output a list of text lines
        """
        with open(self.filename) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    def write_list_to_file(self,
                           input_list):
        """
        Function to write list to file
        with each list item as one line
        """

        # create dir path if it does not exist
        self.dir_create()

        # if file exists, delete it
        self.filename = self.file_remove_check()

        # write to file
        with open(self.filename, 'w') as fileptr:
            for line in input_list:
                fileptr.write('%s\n' % line)

    def write_slurm_script(self,
                           job_name='pyscript',
                           time_in_mins=60,
                           cpus=1,
                           ntasks=1,
                           mem=2000,
                           array=False,
                           iterations=1,
                           **kwargs):
        """
        Write slurm script for the given parameters
        :param job_name: Name of the SLURM job
        :param time_in_mins: expected run time in minutes
        :param cpus: Number of CPUs requested
        :param ntasks: Number of tasks
        :param mem: Memory requested (in MB)
        :param array: (bool) If using job arrays
        :param iterations: Job array upper limit (e.g. 132 for 1-132 array)
        :param kwargs: key word arguments
        :return:
        """
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

        self.write_list_to_file(script_list)

    def write_numpy_array_to_file(self,
                                  np_array,
                                  headers=None):
        """
        Write numpy array to file
        :param np_array: Numpy 2d array to be written to file
        """

        if np_array.dtype.names is not None:
            colnames = list(np_array.dtype.names)
            np.savetxt(self.filename, np_array, delimiter=",", fmt='%3.9f',
                       header=','.join(colnames), comments='')
        else:
            np.savetxt(self.filename, np_array, delimiter=",", fmt='%3.9f',
                       comments='')

    def read_from_csv(self):
        """
        Read csv as list of lists with header.
        Each row is a list and a sample point.
        :param: csv file
        :returns: dictionary of data
        """
        dataframe = pd.read_csv(self.filename, sep=',')

        # convert col names to list of strings
        names = dataframe.columns.values.tolist()

        # convert pixel samples to list
        features = dataframe.values.tolist()
        return {
            'feature': features,
            'name': names,
        }

    def find_all(self, pattern):
        """Find all the names that match pattern"""

        result = []
        # search for a given pattern in a folder path
        for root, dirs, files in os.walk(self.dirname):
            for name in files:
                if fnmatch.fnmatch(name, '*' + pattern + '*'):
                    result.append(os.path.join(root, name))
        return result  # list


class Opt:
    """
    Class to handle notices
    """

    @staticmethod
    def print_memory_usage():
        """
        Function to print memory usage of the python process
        :return print to console/output
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

    @staticmethod
    def time_now():
        """
        Prints current time
        :return: print to console/output
        """
        print('')
        print('*******************************************')
        print('CURRENT TIME: ' + str(datetime.datetime.now()))
        print('*******************************************')
        print('')


class FTPHandler(Handler):
    """Class to handle remote IO for ftp connection"""
    def __init__(self,
                 filename=None,
                 basename=None,
                 dirname=None,
                 ftpserv=None,
                 ftpuser=None,
                 ftppath=None,
                 ftppasswd=None,
                 ftpfilepath=None):

        super(FTPHandler, self).__init__(filename,
                                         basename,
                                         dirname)
        self.ftpserv = ftpserv
        self.ftppath = ftppath
        self.ftpuser = ftpuser
        self.ftppasswd = ftppasswd
        self.ftpfilepath = ftpfilepath
        self.conn = None

    def __repr__(self):
        if self.ftpserv is not None:
            return '<Handler for remote ftp {}>'.format(self.ftpserv)
        else:
            return '<Handler for remote ftp ____>'

    def connect(self):
        """Handle for ftp connections"""
        # define ftp
        ftp = ftplib.FTP(self.ftpserv)

        # login
        if self.ftpuser is not None:
            if self.ftppasswd is not None:
                ftp.login(user=self.ftpuser, passwd=self.ftppasswd)
            else:
                ftp.login(user=self.ftpuser)
        else:
            ftp.login()

        # get ftp connection object
        self.conn = ftp

    def disconnect(self):
        """close connection"""
        self.conn.close()

    def getfiles(self):
        """Copy all files from FTP that are in a list"""

        # connection
        ftp = self.conn

        # get file(s) and write to disk
        if isinstance(self.ftpfilepath, list):
            for i in range(0, len(self.ftpfilepath)):
                with open(self.filename[i], 'wb') as f:
                    ftp.retrbinary("RETR {0}".format(self.ftpfilepath[i]), f.write)
        else:
            with open(self.filename, 'wb') as f:
                ftp.retrbinary("RETR {0}".format(self.ftpfilepath), f.write)

