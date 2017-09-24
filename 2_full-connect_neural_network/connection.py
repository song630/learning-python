from node import Node
from const_node import ConstNode


class Connection:
    def __init__(self, up_node, down_node, weight=0):
        self.up_node = up_node
        self.down_node = down_node
        self.weight = weight

    def __str__(self):
        return 'W{}{}: weight = {}'.format(self.down_node, self.up_node, self.weight)

    def __repr__(self):
        return self.__str__()

    def update_weight(self, rate):  # requires reference of the input node
        self.weight += rate * self.down_node.delta * self.up_node.output
