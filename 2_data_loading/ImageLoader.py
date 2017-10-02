from Loader import Loader


class ImageLoader(Loader):
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16  # + 16 ?
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.cast_to_int(content[start + i * 28 + j]))  # call func of base class
        return picture  # 2-dimension list

    @staticmethod
    def get_one_sample(self, picture):  # convert 2-dimension list to 1-dimension list
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample  # a 784-element list

    def load(self):
        content = self.get_file_content()  # call func of basic class
        data_set = []
        for index in range(self.count):  # member of basic class
            # ===== cannot use "self.get_one_sample()", since it is declared as static =====
            data_set.append(get_one_sample(self.get_picture(content, index)))
        return data_set  # a list of 784-elements lists
