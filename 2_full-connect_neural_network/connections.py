from connection import *


class Connections:
    def __init__(self):
        self.connections = []

    def add_connections(self, connection):  # connection is a list
        for i in connection:
            if i not in self.connections:
                self.connections.append(i)

    def __str__(self):
        for i in self.connections:
            print(i)  # call connection.__str__()
        return ''
