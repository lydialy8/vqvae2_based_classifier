import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import os
from functools import partial
import transforms_mod as transforms


def plot_loss(train_recon_errors, train_perplexities):
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_recon_errors)
    ax.set_yscale('log')
    ax.set_title('NMSE.')
    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_perplexities)
    ax.set_title('Average codebook usage (perplexity).')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return 0  # parts[-2]


def parse_path_overhead(channel, size, band, cnls, mode_4on13, record):
    # load the raw data from the file as a string
    img = tf.io.read_file(record)
    data = tf.io.decode_jpeg(img, channels=3)
    img = data / 255
    select_img = tf.expand_dims(img[:, :, band], axis=2)
    return select_img, img, record

def parse_path(channel, size, band, cnls, mode_4on13, record):
    label = get_label(record)
    # load the raw data from the file as a string
    img = tf.io.read_file(record)
    if cnls ==4:
        nb = 168#858#144#168
    else:
        nb = 1653
    data = tf.io.decode_raw(img, tf.uint16)[nb:]
    img = data * 2 / tf.reduce_max(data) - 1
    reshaped_img = tf.reshape(img, (size, size, cnls))

    if (mode_4on13) or (channel == 4 and cnls == 13) :
        select_img = tf.gather(reshaped_img, [1, 2, 3, 7], axis=2)
    if channel == 1:
        select_img = tf.expand_dims(reshaped_img[:, :, band], axis=2)
    return select_img, reshaped_img, record



def data_generator(files, batch_size, channels=13, band=0, res=512,src_cnl=13,mode_4on13 = False):
    data_augmentation = transforms.Compose([
        transforms.GaussianBlur(kernel_size=9, p=0.2),
        #transforms.RandomCrop(res),
        transforms.Resize(res),
        transforms.RandomFlip(p=0.3),
        transforms.RandomShift(max_percent=0.1, p=0.3),
        transforms.RandomRotation(30, p=0.2),
        transforms.ToTensor()
    ])
    dataset = tf.data.Dataset.list_files(files, shuffle=True)
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(partial(parse_path_overhead, channels, res, band, src_cnl, mode_4on13), num_parallel_calls=autotune)
    dataset = dataset.map(lambda x, y: (tf.py_function(data_augmentation, inp=[x], Tout=tf.float32), y),
                          num_parallel_calls=autotune)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset


def data_generator_val(files, batch_size, channels=13, band=1, res=512, src_cnl=13, mode_4on13=False):
    data_augmentation = transforms.Compose([
        transforms.Resize(res),
        transforms.ToTensor()
    ])
    dataset = tf.data.Dataset.list_files(files, shuffle=False)
    autotune = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(partial(parse_path_overhead, channels, res, band, src_cnl, mode_4on13), num_parallel_calls=autotune)
    #dataset = dataset.map(lambda x, y: (tf.py_function(data_augmentation, inp=[x], Tout=tf.float32), y), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(autotune)
    return dataset