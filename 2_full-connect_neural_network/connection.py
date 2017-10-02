import random
from node import Node


class Connection:
    def __init__(self, up_node, down_node):
        self.up_node = up_node
        self.down_node = down_node
        self.gradient = 0.0  # +++++ used for gradient check +++++
        # ===== if set weight = 0, delta and weight will always be 0 =====
        self.weight = random.uniform(-0.1, 0.1)

    def __str__(self):
        return 'W[{}][{}]: weight = {:.5f}'.format(self.down_node.node_index, self.up_node.node_index, self.weight)

    def __repr__(self):
        return self.__str__()

    # ================================
    def calc_gradient(self):  # +++++ used for gradient check +++++
        self.gradient = self.down_node.delta * self.up_node.output

    def get_gradient(self):  # +++++ used for gradient check +++++
        return self.gradient
    # ================================

    def update_weight(self, rate):  # requires reference of the input node
        self.weight += rate * self.down_node.delta * self.up_node.output
