import fnmatch
from common import Handler
import subprocess
import io
import os
import sys
from google.cloud import storage


__all__ = ['Storage']


# class storage to interact with google cloud storage
class Storage(Handler):

    def __init__(self,
                 name,
                 project=None,
                 credentials=None,
                 input_path=None,
                 output_path=None,
                 dirpath=None):

        """
        Initializes a storage object to interact with Google cloud storage
        :param name: Name of the bucket
        :param project: Name of the project
        :param credentials: Credentials to connect to cloud storage (will check later)
        :param input_path: Emulated folder on bucket
        :param dirpath: Path of file on disk
        """

        super(Storage, self).__init__(dirpath)

        self.name = name
        self.project = project
        self.input_path = input_path
        self.output_path = output_path
        self.dirpath = dirpath
        self.credentials = credentials
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.name)

    def __repr__(self):
        """
        Representation of the storage object
        :return: Representation string
        """
        if self.input_path is not None and self.output_path is not None:
            return "<Storage object for {} with I/O {} {}>".format(self.name,
                                                                   self.input_path,
                                                                   self.dirpath)
        else:
            return "<Storage object for {}>".format(self.name)

    def create_bucket(self):
        """
        Creates a new bucket
        """
        bucket = self.client.create_bucket(self.name)

    def delete_bucket(self):
        """
        Deletes a bucket. The bucket must be empty.
        """
        self.bucket.delete()

    def get_bucket_labels(self):
        """
        Prints out a bucket's labels.
        """
        return self.bucket.labels

    def add_bucket_label(self,
                         key,
                         value):
        """
        Add a label to a bucket.
        :param key: Label key
        :param value: label value
        """
        labels = self.bucket.labels
        labels[key] = value
        self.bucket.labels = labels
        self.bucket.patch()

    def remove_bucket_label(self,
                            key):
        """
        Remove a label from a bucket.
        :param key: Label name (key)
        """
        labels = self.bucket.labels
        if key in labels:
            del labels[key]
        self.bucket.labels = labels
        self.bucket.patch()

    def list_blobs(self,
                   path=None,
                   pattern=None,
                   sort_by_name=False,
                   gsutil=False,
                   get_all=False,
                   files_only=True,
                   **kwargs):
        """
        Lists all the blobs in the bucket that begin with the prefix.
        This can be used to list all blobs in a "folder", e.g. "public/".
        The delimiter argument can be used to restrict the results to only the
        "files" in the given "folder". Without the delimiter, the entire tree under
        the prefix is returned.
        :param path: path of the GCS emulated folder containing blobs
        :param pattern: pattern to search for in the blobs
        :param sort_by_name: flag to sort the blob names alphabetically
        :param get_all: If all results: files and folders in subdirectories should be returned
        :param files_only: If only files should be returned not folders
        :param kwargs: additional key word arguments
        :param gsutil: Should gsutil be used? (default: False)
        :return: list of blob names
        """

        # define delimiter
        delimiter = '/'

        if gsutil:
            if get_all:
                proc = subprocess.Popen(['gsutil', 'ls', '-r', 'gs://{}/{}'.format(self.name,
                                                                                   self.input_path)],
                                        stdout=subprocess.PIPE)
            else:
                proc = subprocess.Popen(['gsutil', 'ls', 'gs://{}/{}'.format(self.name,
                                                                             self.input_path)],
                                        stdout=subprocess.PIPE)
            filestring = proc.stdout.read()
            temp_list = filestring.split('\n')
            name_list = [elem.split('gs://' + self.name + delimiter)[-1] for elem in temp_list if len(elem) > 1]

            if files_only:
                name_list = list(string for string in name_list if string[-1] is not delimiter)

        else:
            # if path is provided
            if path is None:
                path = self.input_path

            # add delimiter at the end of path
            # string if not there
            if path[-1] is not delimiter:
                path = path + delimiter

            # get blobs iterator
            blobs = self.bucket.list_blobs(prefix=path, delimiter=delimiter)

            # get file names from blob names
            name_list = list(blob.name.encode('ascii').split(delimiter)[-1] for blob in blobs
                             if blob.name.encode('ascii')[-1] is not delimiter)

        if len(name_list) > 0:

            # look for a specific pattern if provided
            if pattern is not None:
                pattern_list = list()
                for name in name_list:
                    if fnmatch.fnmatch(name, '*' + pattern + '*'):
                        pattern_list.append(name)

                # sort by name
                if sort_by_name:
                    result = list(t for t in sorted(pattern_list))

                else:
                    result = pattern_list

            else:

                # sort by name
                if sort_by_name:
                    result = list(t for t in sorted(name_list))

                # no sort
                else:
                    result = name_list
        else:
            result = []

        return result

    def multi_pattern_list(self,
                           path=None,
                           pattern=None,
                           sort_by_name=False,
                           gsutil=False):
        """
        Lists all the blobs in the bucket that have a list of suffixes.
        This method is a bit slower but a working solution for multiple searches
        :param path: path of the GCS emulated folder containing blobs
        :param pattern: pattern to search for in the blobs
        :param sort_by_name: flag to sort the blob names alphabetically
        :param gsutil: Should gsutil be used? (default: False)
        :return: list of blob names
        """
        if isinstance(pattern, list):
            result = list()
            for i in range(0, len(pattern)):
                temp = self.list_blobs(path=path,
                                       pattern=pattern[i],
                                       sort_by_name=sort_by_name,
                                       gsutil=gsutil)
                if isinstance(temp, list):
                    for j in range(0, len(temp)):
                        result.append(temp[j])
                elif temp is not None:
                    result.append(temp)
                else:
                    pass

            out_result = list()
            for i in result:
                if i not in out_result:
                    out_result.append(i)

            return out_result
        else:
            return self.list_blobs(path=path,
                                   pattern=pattern,
                                   sort_by_name=sort_by_name,
                                   gsutil=gsutil)

    def upload_blob(self,
                    source_file,
                    destination_blob_name,
                    from_memory=False):
        """
        Uploads a file to the bucket from file on disk or from memory
        :param source_file: File path on disk or BytesIO memory buffer
        :param destination_blob_name: File name on bucket
        :param from_memory: Flag to read file from memory buffer (TRUE) or disk (FALSE)
        :return: NoneType
        """

        if from_memory:
            blob = self.bucket.blob(self.output_path + '/' + destination_blob_name)
            url = blob._get_download_url()
            blob.upload_from_string(source_file,
                                    client=self.client)
            return url
        else:
            blob = self.bucket.blob(self.output_path + '/' + destination_blob_name)
            url = blob._get_download_url()
            blob.upload_from_filename(source_file)
            return url

    def upload_blobs(self,
                     file_list,
                     destination_namelist=None,
                     from_memory=False):
        """
        Method to upload multiple
        :param file_list: List of file names on disk or list of ByteIO buffers in memory
        :param destination_namelist: List of destination blob names
        :param from_memory: Flag to read file from memory (TRUE) or disk (FALSE)
        :return: NoneType
        """
        _DEFAULT_CONTENT_TYPE = u'application/octet-stream'
        _STRING_CONTENT_TYPE = u'text/plain'

        url_list = list()

        if not from_memory:

            for i, in_file in enumerate(file_list):

                with open(in_file, 'rb') as file_obj:

                    total_bytes = os.fstat(file_obj.fileno()).st_size

                    if destination_namelist is not None:
                        blob = self.bucket.blob(self.output_path + '/' + destination_namelist[i])
                    else:
                        blob = self.bucket.blob(self.output_path + '/' + Handler(in_file).basename)

                    url_list.append(blob._get_download_url())

                    blob.upload_from_file(file_obj=file_obj,
                                          content_type=_DEFAULT_CONTENT_TYPE,
                                          client=self.client,
                                          size=total_bytes,
                                          predefined_acl=None)
        else:

            for i, file_buffer in enumerate(file_list):

                if destination_namelist is not None:
                    blob = self.bucket.blob(self.output_path + '/' + destination_namelist[i])
                else:
                    raise ValueError("Blob name(s) not found: Upload cannot be completed")

                url_list.append(blob._get_download_url())

                data_size = len(file_buffer.getvalue())

                blob.upload_from_file(file_obj=file_buffer,
                                      size=data_size,
                                      content_type=_STRING_CONTENT_TYPE,
                                      client=self.client,
                                      predefined_acl=None)

        return url_list

    def download_blob(self,
                      source_blob,
                      destination_file=None,
                      in_memory=False):
        """
        Downloads a blob from the bucket.
        :param source_blob: Full file name on bucket
        :param destination_file: File name on disk
        :param in_memory: Should the file be downloaded in memory (default: False)
        :return: File output path if writing to file, memory pointer if in memory
        """
        if self.input_path[0] == '/':
            self.input_path = self.input_path[1:]

        if self.input_path[-1] == '/':
            self.input_path = self.input_path[:-1]

        # define blob object
        blob = self.bucket.blob(self.input_path + '/' + source_blob)

        # perform retrieval
        if in_memory:
            mem_buffer = io.BytesIO()
            blob.download_to_file(mem_buffer)
            return mem_buffer

        else:
            blob.download_to_filename(self.dirpath + self.sep + destination_file)
            return self.dirpath + self.sep + destination_file

    def download_blobs(self,
                       source_blobs,
                       urls=None,
                       in_memory=False):
        """
        Downloads a blob from the bucket.
        :param source_blobs: List of full file name on bucket
        :param urls: List of corresponding GCS URLs for list of source_blobs
        :param in_memory: Should the file be downloaded in memory (default: False)
        :return: List of output paths if writing to file, List of memory pointers if in memory
        """
        transport = self.client._http

        output = list()

        for i, source_blob in enumerate(source_blobs):

            if self.input_path[0] == '/':
                self.input_path = self.input_path[1:]

            if self.input_path[-1] == '/':
                self.input_path = self.input_path[:-1]

            # define blob object
            blob = self.bucket.blob(self.input_path + '/' + source_blob)

            headers = dict()
            if urls is not None:
                url = urls[i]
            else:
                url = blob._get_download_url()

            # perform retrieval
            if in_memory:

                mem_buffer = io.BytesIO()
                blob._do_download(transport,
                                  mem_buffer,
                                  url,
                                  headers)

                output.append(mem_buffer)

            else:
                outfile = self.dirpath + self.sep + source_blob
                blob._do_download(transport,
                                  outfile,
                                  url,
                                  headers)

                output.append(outfile)

        return output

    def delete_blob(self,
                    blob_name,
                    from_input_path=False):
        """
        Deletes a blob from the bucket.
        :param blob_name: Full file name on bucket (this will include path)
        :param from_input_path: If the blob is in bucket.input_path
        """
        if from_input_path:
            blob = self.bucket.blob(self.input_path + '/' + blob_name)
        else:
            blob = self.bucket.blob(self.output_path + '/' + blob_name)

        blob.delete()

    def get_blob_metadata(self,
                          blob_name):
        """
        Prints out a blob's metadata.
        :param blob_name: Full file name on bucket (this will include path)
        :return:
        """
        blob = self.bucket.get_blob(self.input_path + '/' + blob_name)
        return {
            'name': blob.name,
            'bucket': blob.bucket.name,
            'storage_class': blob.storage_class,
            'id': blob.id,
            'size': blob.size,
            'updated': blob.updated,
            'generation': blob.generation,
            'metageneration': blob.metageneration,
            'etag': blob.etag,
            'owner': blob.owner,
            'component_count': blob.component_count,
            'crc32c': blob.crc32c,
            'md5_hash': blob.md5_hash,
            'cache_control': blob.cache_control,
            'content_type': blob.content_type,
            'content_disposition': blob.content_disposition,
            'content_encoding': blob.content_encoding,
            'content_language': blob.content_language,
            'Metadata': blob.metadata,
        }

    def make_blob_public(self,
                         blob_name):
        """
        Makes a blob publicly accessible.
        :param blob_name: Full file name on bucket (this will include path)
        """
        blob = self.bucket.blob(self.output_path + '/' + blob_name)
        blob.make_public()

    def generate_url(self,
                     blob_name):
        """
        Generates download URL for a blob.
        :param blob_name: File name on bucket (this does not include path)
        :return: URL for a GCS blob
        """
        blob = self.bucket.blob(self.input_path + '/' + blob_name)
        return str(blob._get_download_url())

    def rename_blob(self,
                    blob_name,
                    new_name):
        """
        Renames a blob.
        :param blob_name: Full file name on bucket (this will include path)
        :param new_name: New full file name on bucket (this will include path)
        """
        blob = self.bucket.blob(self.output_path + '/' + blob_name)
        new_blob = self.bucket.rename_blob(self.output_path + '/' + blob, self.output_path + '/' + new_name)

    def copy_blob(self,
                  blob_name,
                  new_blob_name,
                  new_bucket_name=None,
                  new_path=None):
        """
        Copies a blob from one bucket to another with a new name.
        :param blob_name: Name of the current blob (current file in GCS)
        :param new_blob_name: Name of the new file (destination file in GCS)
        :param new_bucket_name: Name of the new bucket to copy blob to (if any)
        :param new_path: Output path for the copy
        :return:
        """
        source_blob = self.bucket.blob(self.input_path + '/' + blob_name)

        if new_bucket_name is not None:
            destination_bucket = Storage(new_bucket_name)
            destination_bucket.output_path = new_path
            new_blob = self.bucket.copy_blob(source_blob,
                                             destination_bucket.bucket,
                                             destination_bucket.output_path + '/' + new_blob_name)
        elif new_path is not None:
            self.output_path = new_path
            self.bucket.copy_blob(source_blob,
                                  self.bucket,
                                  self.output_path + '/' + new_blob_name)

        else:
            self.bucket.copy_blob(source_blob,
                                  self.bucket,
                                  self.output_path + '/' + new_blob_name)

    @staticmethod
    def copy_folder(bucket_name=None,
                    bucket_path=None,
                    disk_path=None,
                    to_bucket=False,
                    verbose=False):

        """
        Method to copy contents of a folder to or from bucket
        :param bucket_name: Name of the bucket
        :param bucket_path: Path to copy (to), example:
        :param disk_path: Folder on disk to copy
        :param to_bucket: if copying to bucket
        :param verbose: If some steps should be displayed
        :return: None
        """

        if None in [bucket_name, bucket_path, disk_path]:
            raise ValueError('Missing input value')
        else:
            if to_bucket:
                proc = subprocess.Popen(['gsutil', '-m', 'cp', '-R',
                                         disk_path + '/*',
                                         'gs://{}/{}/'.format(bucket_name,
                                                              bucket_path)],
                                        stdout=subprocess.PIPE)
            else:
                proc = subprocess.Popen(['gsutil', '-m', 'cp', '-R',
                                         'gs://{}/{}/*'.format(bucket_name,
                                                               bucket_path),
                                         disk_path],
                                        stdout=subprocess.PIPE)

            filestring = proc.stdout.read()
            if verbose:
                sys.stdout.write(filestring + '\n')
