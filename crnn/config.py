from pprint import pprint


class Config:

    # data
    train_filename = ""
    val_filename = ""
    root_dir = ""

    char_dict_file = ""
    image_size = (32, 100)
    max_label_length = 10

    # cuda
    device = "cuda:0"

    # network
    nclasses = 5990

    # training
    epoch = 100


opt = Config()
