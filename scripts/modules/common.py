import os
import numpy as np
import pandas as pd
import datetime
import fnmatch
import psutil
import ftplib
import sys
import gzip
import glob

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
        """Find size of a list object even if it is a one element non-list"""

        if isinstance(query_list, list):
            return len(query_list)
        else:
            return 1

    @staticmethod
    def custom_list(start, end, step=None):
        """
        List with custom first number but rest all follow same increment; integers only
        :param start: starting integer
        :param end: ending integer
        :param step: step integer
        :return: list of integers
        """

        # end of list
        end = end + step - (end % step)

        # initialize the list
        out = list()

        # make iterator
        if step is not None:
            if start % step > 0:
                out.append(start)
                if start < step:
                    start = step
                elif start > step:
                    start = start + step - (start % step)
            iterator = range(start, end, step)
        else:
            step = 1
            iterator = range(start, step, end)

        # add the rest of the list to out
        for i in iterator:
            out.append(i)

        return out

    def sublistfinder(self, pattern):
        """
        Find the location of sub-list in a python list, no repetitions in mylist or pattern
        :param pattern: shorter list
        :return: list of locations of elements in mylist ordered as in pattern
        """
        return [self.index(x) for x in pattern if x in self]

    def count_in_range(self, llim, ulim):
        """
        Find elements in a range
        :param ulim: upper limit
        :param llim: lower limit
        :return: count, list
        """
        return len([i for i in self if llim <= i < ulim])

    @staticmethod
    def frange(start, end, step):
        """
        To make list from float arguments
        :param start: start number
        :param end: end number
        :param step: step
        :return: list
        """
        if end > start:
            if (end-start) % step > 0.0:
                n = long((end-start) / step) + 2
            else:
                n = long((end - start) / step) + 1

            temp = np.zeros(n, dtype=float)

            for i in range(0, n - 1):
                temp[i] = start + i * step

            temp[n - 1] = end

            return list(temp)

        else:
            raise ValueError("Start value is less than end value")

    def where(self, operator='=', obj=None):
        """
        Locate the list of indices given a condition
        :param operator: boolean operator (options: =, <, >, =<, ,=>, !=)
        :param obj: Object to compare with input list
        :return: List of indices
        """
        if operator == '=':
            loc = list(i for i in range(0, len(self)) if self[i] == obj)
        elif operator == '>':
            loc = list(i for i in range(0, len(self)) if self[i] > obj)
        elif operator == '<':
            loc = list(i for i in range(0, len(self)) if self[i] < obj)
        elif operator == '<=':
            loc = list(i for i in range(0, len(self)) if self[i] <= obj)
        elif operator == '>=':
            loc = list(i for i in range(0, len(self)) if self[i] >= obj)
        elif operator == '!=':
            loc = list(i for i in range(0, len(self)) if self[i] != obj)
        else:
            loc = None

        if len(loc) == 1:
            return loc[0]
        else:
            return loc

    @staticmethod
    def column(matrix, i):
        """
        Get column of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[:, i].tolist()
        return list(elem[0] for elem in mat)

    @staticmethod
    def row(matrix, i):
        """
        Get row of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[i].tolist()
        return mat[0]


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
        except Exception:
            self.basename = basename

        try:
            self.dirname = os.path.dirname(filename)
        except Exception:
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
            pass  # print('Path already exists: ' + self.dirname)

    def add_to_filename(self, string):
        components = self.basename.split('.')
        if len(components) >= 2:
            return self.dirname + self.sep + '.'.join(components[0:-1]) + \
                      string + '.' + components[-1]
        else:
            return self.basename + self.sep + components[0] + string

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

    def extract_gz(self):
        """
        Extract landsat file to a temp folder
        """

        # define a temp folder with scene ID
        tempfile = self.dirname + self.sep + self.basename.split('.gz')[0]

        Handler(tempfile).file_remove_check()

        # extract infile to temp folder
        if self.filename.endswith(".gz"):
            with gzip.open(self.filename, 'rb') as gf:
                file_content = gf.read()
                with open(tempfile, 'wb') as fw:
                    fw.write(file_content)

        else:  # not a tar.gz archive
            raise TypeError("Not a .gz archive")

    def write_list_to_file(self,
                           input_list,
                           rownames=None,
                           colnames=None,
                           delim=", "):
        """
        Function to write list to file
        with each list item as one line
        :param input_list: input text list
        :param rownames: list of row names strings
        :param colnames: list of column name strings
        :param delim: delimiter (default: ", ")
        :return: write to file
        """
        print('inp list:')
        print(input_list)
        print(colnames)
        print(rownames)

        # add rownames and colnames
        if rownames is not None and colnames is not None:
            input_list = [str(rownames[i]) + delim + input_list[i] for i in range(0, len(input_list))]
            header = delim + ", ".join([str(elem) for elem in colnames])
            input_list = [header] + input_list
        if rownames is None and colnames is not None:
            header = ", ".join([str(elem) for elem in colnames])
            input_list = [header] + input_list
        if rownames is not None and colnames is None:
            input_list = [rownames[i] + delim + input_list[i] for i in range(0, len(input_list))]

        print('inp list:')
        print(input_list)
        print(colnames)
        print(rownames)

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
                                  colnames=None,
                                  rownames=None,
                                  delim=", "):
        """
        Write numpy array to file
        :param np_array: Numpy 2d array to be written to file
        :param colnames: list of column name strings
        :param rownames: list of row name strings
        :param delim: Delimiter (default: ", ")
        """

        # format numpy 2d array as list of strings, each string is a line to be written
        inp_list = [delim.join(["{:{w}.{p}f}".format(np_array[i, j], w=4, p=9)
                                for j in range(0, np_array.shape[1])])
                    for i in range(0, np_array.shape[0])]

        # if column names are available from numpy array, this overrides input colnames
        if np_array.dtype.names is not None:
            colnames = list(np_array.dtype.names)

        # write to file
        self.write_list_to_file(inp_list,
                                rownames=rownames,
                                colnames=colnames,
                                delim=delim)

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

        Opt.cprint('')
        Opt.cprint('*******************************************')
        Opt.cprint(print_str)
        Opt.cprint('*******************************************')
        Opt.cprint('')

    @staticmethod
    def time_now():
        """
        Prints current time
        :return: print to console/output
        """
        Opt.cprint('')
        Opt.cprint('*******************************************')
        Opt.cprint('CURRENT TIME: ' + str(datetime.datetime.now()))
        Opt.cprint('*******************************************')
        Opt.cprint('')

    @staticmethod
    def cprint(text):
        sys.stdout.write(str(text) + '\n')
        sys.stdout.flush()


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
        ftp_conn = self.conn

        # get file(s) and write to disk
        if isinstance(self.ftpfilepath, list):
            self.filename = [self.dirname + Handler().sep + Handler(ftpfile).basename
                             for ftpfile in self.ftpfilepath]

            for i in range(0, len(self.ftpfilepath)):
                with open(self.filename[i], 'wb') as f:
                    try:
                        if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath[i]), f.write):
                            print('Copying file {} to {}'.format(Handler(self.ftpfilepath[i]).basename,
                                                                 self.dirname))

                    except:
                        print('File {} not found or already written'.format(Handler(self.ftpfilepath[i]).basename))
        else:
            self.filename = self.dirname + Handler().sep + Handler(self.ftpfilepath).basename
            with open(self.filename, 'wb') as f:
                try:
                    if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath), f.write):
                        print('Copying file {} to {}'.format(Handler(self.ftpfilepath).basename,
                                                             self.dirname))
                except:
                    print('File {} not found or already written'.format(Handler(self.ftpfilepath).basename))

