import struct


class Loader:
    def __init__(self, file_path, count):
        """
        :param file_path: path of .gz files
        :param count: num of samples or labels
        """
        self.path = file_path
        self.count = count

    def get_content(self):  # get all the data of file (in bytes)
        file = open(self.path, 'rb')
        content = file.read()
        file.close()
        return content

    @staticmethod  # no "self" appeared in function
    def cast_to_int(self, byte):  # convert an unsigned byte char to an int
        return struct.unpack('B', byte)[0]
