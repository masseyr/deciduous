from decimal import *
import csv
import numpy as np
import datetime
import fnmatch
import random
import psutil
import ftplib
import time
import copy
import gzip
import sys
import os


__all__ = ['Sublist',
           'Handler',
           'FTPHandler',
           'Opt']


class Sublist(list):
    """
    Class to handle list operations
    """
    def __eq__(self,
               other):
        """
        Check for a = other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] == other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __gt__(self,
               other):
        """
        Check for a > other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] > other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __ge__(self,
               other):
        """
        Check for a >= other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] >= other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __lt__(self,
               other):
        """
        Check for a < other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] < other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __le__(self,
               other):
        """
        Check for a <= other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] <= other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __ne__(self,
               other):
        """
        Check for a != other
        return: List of indices
        """
        temp = list(i for i in range(0, len(self)) if self[i] != other)

        if len(temp) == 1:
            return temp[0]
        else:
            return temp

    def __getitem__(self,
                    item):
        """
        Method to get item(s) from the list using list or number as index
        :param item: Number or list of numbers
        :return: List
        """

        try:
            if isinstance(item, list):
                return list(list(self)[i] for i in item)
            else:
                return list(self)[item]

        except (TypeError, KeyError):
            print("List index not a number or list of numbers")

    def range(self,
              llim,
              ulim,
              index=False):
        """
        Make a subset of the list by only including elements between ulim and llim.
        :param ulim: upper limit
        :param llim: lower limit
        :param index: If the function should return index of values that lie within the limits
        :return: List
        """
        if index:
            return [i for i, x in enumerate(self) if llim <= x <= ulim]
        else:
            return [x for x in self if llim <= x <= ulim]

    def add(self,
            elem):
        """
        Add two lists
        :param elem: Another list or element
        :return: list
        """
        if isinstance(elem, list):
            for val in elem:
                self.append(val)
            return self
        else:
            self.append(elem)
            return self

    def remove_by_loc(self,
                      elem):
        """
        Method to remove a Sublist element with index 'other'
        :param elem: Index or list of indices
        :return: Sublist
        """
        if isinstance(elem, list):
            return list(val for j, val in enumerate(self) for loc in elem if j != loc)
        else:
            return list(val for j, val in enumerate(self) if j != elem)

    def remove(self,
               elem):
        """
        Method to remove list item or sublist
        :param elem: item or list
        :return: list
        """
        if isinstance(elem, list):
            return list(val for val in self if val not in elem)
        else:
            return list(val for val in self if val != elem)

    @staticmethod
    def list_size(query_list):
        """
        Find size of a list object even if it is a one element non-list
        :param query_list: List to be sized

        """

        if isinstance(query_list, list):
            return len(query_list)
        else:
            return 1

    def remove_by_percent(self,
                          percent):
        """
        Method to remove randomly selected elements by percentage in a list
        :param percent: Percentage (Range 0-100)
        :return: list
        """
        nelem = len(self)
        nelem_by_percent = int(round((float(nelem)*float(100 - percent))/float(100)))
        return random.sample(self, nelem_by_percent)

    def tuple_by_pairs(self):
        """
        Make a list of tuple pairs of consequetive list elements
        :return: List of tuples
        """

        return Sublist((self[i], self[i+1]) for i in range(len(self) - 1))

    @classmethod
    def custom_list(cls,
                    start,
                    end,
                    step=None):
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
        out = Sublist()

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

    def sublistfinder(self,
                      pattern):
        """
        Find the location of sub-list in a python list, no repetitions in mylist or pattern
        :param pattern: shorter list
        :return: list of locations of elements in mylist ordered as in pattern
        """
        return Sublist(self.index(x) for x in pattern if x in self)

    def count_in_range(self,
                       llim,
                       ulim):
        """
        Find elements in a range
        :param ulim: upper limit
        :param llim: lower limit
        :return: count, list
        """
        return len([i for i in self if llim <= i < ulim])

    @classmethod
    def frange(cls,
               start,
               end,
               step):
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

            return Sublist(temp)

        else:
            raise ValueError("Start value is less than end value")

    @classmethod
    def column(cls,
               matrix,
               i):
        """
        Get column of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[:, i].tolist()
        return Sublist(elem[0] for elem in mat)

    @classmethod
    def row(cls,
            matrix,
            i):
        """
        Get row of a numpy matrix
        :param matrix: Numpy matrix
        :param i: index
        :return: List
        """
        mat = matrix[i].tolist()
        return Sublist(mat[0])

    def mean(self):
        """
        calculate mean of an array
        :return: float
        """
        return np.mean(self)

    def median(self):
        """
        calculate median of an array
        :return: float
        """
        return np.median(self)

    @staticmethod
    def percentile(arr, pctl=95.0):
        """
        Method to output percentile in an array
        :param arr: input numpy array or iterable
        :param pctl: percentile value
        :return:
        """

        pctl_calc = np.percentile(arr, pctl)
        diff = np.abs(np.array(arr) - pctl_calc)
        closest_vals = np.array(arr)[diff == min(diff)]
        if len(closest_vals) > 1:
            return min(closest_vals)
        else:
            return closest_vals.item()

    @staticmethod
    def pctl_interval(arr, intvl=95.0):
        """
        Method to calculate the width of a percentile interval
        :param arr: input numpy array or iterable
        :param intvl: Interval to calculate (default: 95th percentile)
        :return: scalar
        """

        lower = Sublist.percentile(arr, (100.0 - intvl)/2.0)
        upper = Sublist.percentile(arr, 100.0 - (100.0 - intvl)/2.0)

        return np.abs((upper - lower)/2.0)


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
        csv_data = np.genfromtxt(self.filename,
                                 delimiter=',',
                                 names=True,
                                 encoding="utf-8",
                                 dtype=None)

        names = [csv_data.dtype.names[i] for i in range(0, len(csv_data.dtype.names))]
        return names, csv_data

    def read_text_by_line(self):
        """
        Read text file line by line and output a list of text lines
        """
        with open(self.filename) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    def read_array_from_csv(self,
                            delim=",",
                            array_1d=False,
                            nodataval=None):
        """
        Read array from file and output a numpy array. There should be no header.
        :param delim: Delimiter (default ',')
        :param array_1d: Should the array be reshaped to 1-dimensional array? (default: False)
        :param nodataval: Data values to be removed from 1-dimensional array
                          (ignored if array_1d flag is false)
        :returns: Numpy array (2d or 1d)
        """
        with open(self.filename) as f:
            content = f.readlines()
            lines = [x.strip() for x in content]
            ncols = len(lines[0].split(delim))
            nrows = len(lines)
            arr = np.zeros((nrows, ncols))
            for i, line in enumerate(lines):
                arr[i:] = [float(elem.strip()) for elem in line.split(delim)]

        if array_1d:
            lenr, lenc = arr.shape
            arr_out = arr.reshape(lenr * lenc)
            if nodataval is not None:
                arr_out = arr_out[arr_out != nodataval]
            return arr_out
        else:
            return arr

    def extract_gz(self,
                   dirname=None,
                   add_ext=None):
        """
        Extract landsat file to a temp folder
        :param dirname: Folder to extract all files to
        :param add_ext: Extension to be added to extracted file
        """

        if add_ext is None:
            add_ext = ''
        else:
            add_ext = '.' + str(add_ext)

        # define a temp file with scene ID
        if dirname is not None:
            tempfile = dirname + self.sep + self.basename.split('.gz')[0] + add_ext
        else:
            tempfile = self.dirname + self.sep + self.basename.split('.gz')[0] + add_ext

        # remove if file already exists
        temp = Handler(tempfile)
        temp.file_remove_check()

        # extract infile to temp folder
        if self.filename.endswith(".gz"):
            Opt.cprint('Extracting {} to {}'.format(self.basename,
                                                    temp.basename))
            with gzip.open(self.filename, 'rb') as gf:
                file_content = gf.read()
                with open(tempfile, 'wb') as fw:
                    fw.write(file_content)

        else:  # not a tar.gz archive
            raise TypeError("Not a .gz archive")

    def get_size(self,
                 unit='kb',
                 precision=3,
                 as_long=True):
        """
        Function to get file size
        :param unit: Unit to get file size in (options: 'b', 'kb', 'mb', 'gb', 'tb', 'pb', 'bit')
        :param precision: Precision to report
        :param as_long: Should the output be returned as a long integer? This truncates any decimal value
        :return: long integer
        """
        size = os.path.getsize(self.filename)

        getcontext().prec = precision

        if unit == 'bit':
            output = float(Decimal(size)*Decimal(2**3))
        elif unit == 'kb':
            output = float(Decimal(size)/Decimal(2**10))
        elif unit == 'mb':
            output = float(Decimal(size)/(Decimal(2**20)))
        elif unit == 'gb':
            output = float(Decimal(size)/(Decimal(2**30)))
        elif unit == 'tb':
            output = float(Decimal(size)/(Decimal(2**40)))
        elif unit == 'pb':
            output = float(Decimal(size)/(Decimal(2**50)))
        else:
            output = size
        if as_long:
            return long(round(output, precision))
        else:
            return round(output, precision)

    def write_list_to_file(self,
                           input_list,
                           rownames=None,
                           colnames=None,
                           delim=", ",
                           append=False):
        """
        Function to write list to file
        with each list item as one line
        :param input_list: input text list
        :param rownames: list of row names strings
        :param colnames: list of column name strings
        :param delim: delimiter (default: ", ")
        :param append: if the lines should be appended to the file
        :return: write to file
        """
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

        # create dir path if it does not exist
        self.dir_create()

        # if file exists, delete it
        self.filename = self.file_remove_check()

        # write to file
        if append:
            open_type = 'a'  # append
        else:
            open_type = 'w'  # write

        with open(self.filename, open_type) as fileptr:
            for line in input_list:
                fileptr.write('{}\n'.format(str(line)))

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
        lines = list()
        for i in range(0, np_array.shape[0]):
            lines.append(delim.join(["{:{w}.{p}f}".format(np_array[i, j], w=4, p=9)
                                     for j in range(0, np_array.shape[1])]))

        # if column names are available from numpy array, this overrides input colnames
        if np_array.dtype.names is not None:
            colnames = list(np_array.dtype.names)

        # write to file
        self.write_list_to_file(lines,
                                rownames=rownames,
                                colnames=colnames,
                                delim=delim)

    def read_from_csv(self,
                      return_dicts=False,
                      num_tries=10,
                      wait_time=2):
        """
        Read csv as list of lists with header.
        Each row is a list and a sample point.
        :param return_dicts: If list of dictionaries should be returned (default: False)
        :param num_tries: Number of tries for reading the csv file (default: 10)
        :param wait_time: Time to wait between tries(default: 2 seconds)
        :returns: dictionary of data
        """
        lines = list()
        tries = 0
        while len(lines) == 0 and tries < num_tries:
            try:
                with open(self.filename, 'r') as ds:
                    for line in csv.reader(ds, delimiter=','):
                        lines.append(line)
            except Exception as e:
                print(e)
                time.sleep(wait_time)
            tries += 1

        # convert col names to list of strings
        names = list(name.strip() for name in lines[0])

        if len(lines) > 0:
            # convert pixel samples to list
            if return_dicts:
                return list(dict(zip(names, list(self.string_to_type(elem) for elem in feat)))
                            for feat in lines[1:])

            else:
                return {
                    'feature': list(list(self.string_to_type(elem) for elem in feat)
                                    for feat in lines[1:]),
                    'name': names,
                }
        else:
            return {
                'feature': list(),
                'name': list(),
            }

    @staticmethod
    def write_to_csv(list_of_dicts,
                     outfile=None,
                     delimiter=',',
                     append=False,
                     header=True):

        if outfile is None:
            raise ValueError("No file name for writing")

        lines = list()
        if header:
            lines.append(delimiter.join(list(list_of_dicts[0])))
        for data_dict in list_of_dicts:
            lines.append(delimiter.join(list(str(val) for _, val in data_dict.items())))

        if append:
            with open(outfile, 'a') as f:
                for line in lines:
                    f.write(line + '\n')
        else:
            with open(outfile, 'w') as f:
                for line in lines:
                    f.write(line + '\n')

    def find_all(self,
                 pattern='*'):
        """
        Find all the names that match pattern
        :param pattern: pattern to look for in the folder
        """
        result = []
        # search for a given pattern in a folder path
        if pattern == '*':
            search_str = '*'
        else:
            search_str = '*' + pattern + '*'

        for root, dirs, files in os.walk(self.dirname):
            for name in files:
                if fnmatch.fnmatch(name, search_str):
                    if str(root) in str(self.dirname) or str(self.dirname) in str(root):
                        result.append(os.path.join(root, name))

        return result  # list

    @staticmethod
    def string_to_type(x):
        """
        Method to return name of the data type
        :param x: input item
        :return: string
        """
        if type(x).__name__ == 'str':
            try:
                val = int(x)
            except ValueError:
                try:
                    val = float(x)
                except ValueError:
                    try:
                        val = str(x)
                    except:
                        val = None
            x = val
        return x


class Opt:
    """
    Class to handle notices
    """
    def __init__(self,
                 obj=None):
        self.obj = obj  # mutable object

    def __repr__(self):
        return "Optional helper and notice class"

    @staticmethod
    def print_memory_usage():
        """
        Function to print memory usage of the python process
        :return print to console/output
        """
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss

        if 2**10 <= mem < 2**20:
            div = float(2**10)
            suff = ' KB'
        elif 2**20 <= mem < 2**30:
            div = float(2**20)
            suff = ' MB'
        elif 2**30 <= mem < 2**40:
            div = float(2**30)
            suff = ' GB'
        elif mem >= 2**40:
            div = float(2**40)
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

    @staticmethod
    def __copy__(obj):
        return copy.deepcopy(obj)


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
            filenamelist = [self.dirname + self.sep + Handler(ftpfile).basename
                            for ftpfile in self.ftpfilepath]

            for i in range(0, len(filenamelist)):
                self.filename = filenamelist[i]
                with open(self.filename, 'wb') as f:
                    try:
                        if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath[i]), f.write):
                            print('Copying file {} to {}'.format(self.basename,
                                                                 self.dirname))

                    except Exception as err:
                        Opt.cprint('File {} not found or already written\n Error: {}'.format(self.basename,
                                                                                             err))
        else:
            self.filename = self.dirname + Handler().sep + Handler(self.ftpfilepath).basename
            with open(self.filename, 'wb') as f:
                try:
                    if ftp_conn.retrbinary("RETR {}".format(self.ftpfilepath), f.write):
                        Opt.cprint('Copying file {} to {}'.format(Handler(self.ftpfilepath).basename,
                                                             self.dirname))
                except Exception:
                    Opt.cprint('File {} not found or already written'.format(self.basename))

