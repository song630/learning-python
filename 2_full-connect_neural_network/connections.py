from connection import *


class Connections:
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):  # connection is a list
        if connection not in self.connections:
            self.connections.append(connection)

    def __str__(self):
        for i in self.connections:
            print(i)  # call connection.__str__()
        return ''
