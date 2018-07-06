class UninitializedError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 500

    def __init__(self, error=None,
                 description="Object not initialized",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class ObjectNotFound(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 404

    def __init__(self, error=None,
                 description="Object not found",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error