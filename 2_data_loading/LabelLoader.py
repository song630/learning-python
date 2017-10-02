from Loader import Loader


class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()  # call func of basic class
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels  # list of normalized lists
    
    def norm(self, label):  # convert an int to a list
        label_vec = []
        label_value = self.cast_to_int(label)  # call func of basic class
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
