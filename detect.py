
import tensorflow as tf
import model
import cv2
import glob

import numpy as np
import sonnet as snt
from model import Encoder, Decoder, VQVAEModel
from utils import *
import os
import pathlib
import tifffile as tiff
from sklearn import metrics
import matplotlib.pyplot as plt


bands = 13
height = width = res = 32
stepSize = 8
thresh =0.00042331408

plt.rcParams.update({'font.size': 16})
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--channels', type=int, default=13, choices=[3, 4, 13],help='number of channels in an image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--steps_count', type=int, default=100000, help='training steps')
    parser.add_argument('--hiddens', type=int, default=256, help='number of hiddens')
    parser.add_argument('--residual_hiddens', type=int, default=64, help='number of residual hidden layers')
    parser.add_argument('--residual_layers', type=int, default=2, help='number of residual layers')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding code dimensions')
    parser.add_argument('--num_embeddings', type=int, default=512, help='capacity of embeddings')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost')
    parser.add_argument('--decay', type=float, default=0.99, help='decay of learning rate')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--save_steps', type=int, default=1000, help='save model rate')
    parser.add_argument('--vq_use_ema', type=str2bool, default=True, help='uses EMA optimizer instead of Adam for codebook updates')
    parser.add_argument('--training_flag', type=str2bool, default=False, help='train or just test')
    parser.add_argument('--reconstruct_images', type=str2bool, default=False, help='save reconstructed images and their relative squared error')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',  help='Directory for saving checkpoints')
    parser.add_argument('--input_dir_test', type=str, default='D:\\styleTransferExperiments\\repositories\\pix2pix_cyclegan_multispectral\\results\\image\\*2_3.tif',  help='Directory input test images')
    parser.add_argument('--output_dir', type=str, default='.\\output_detect',  help='Directory for saving generated images')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set hyper-parameters.
    batch_size = args.batch_size
    image_size = args.image_size
    num_training_updates = args.steps_count
    num_hiddens = args.hiddens
    num_residual_hiddens = args.residual_hiddens
    num_residual_layers = args.residual_layers
    embedding_dim = args.embedding_dim
    save_steps = args.save_steps
    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = args.num_embeddings
    checkpoint_dir = args.checkpoint_dir
    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = args.commitment_cost

    # Use EMA updates for the codebook (instead of the Adam optimizer).
    # This typically converges faster, and makes the model less dependent on choice
    # of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
    # developed afterwards). See Appendix of the paper for more details.
    vq_use_ema = args.vq_use_ema

    # This is only used for EMA updates.
    decay = args.decay

    learning_rate = args.lr

    # # Build modules.
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,1)
    encoder2 = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,2)
    encoder3 = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,3)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,3)
    decoder2 = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,2)
    decoder3 = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,1)



    pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                              kernel_shape=(1, 1),
                              stride=(1, 1),
                              name="to_vq")

    pre_vq_conv2 = snt.Conv2D(output_channels=embedding_dim,
                                    kernel_shape=(1, 1),
                                    stride=(1, 1),
                                    name="to_vq_2")

    pre_vq_conv3 = snt.Conv2D(output_channels=embedding_dim,
                                    kernel_shape=(1, 1),
                                    stride=(1, 1),
                                    name="to_vq_2")

    if vq_use_ema:
        vq_vae = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)
        vq_vae2 = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)

        vq_vae3 = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)
    else:
        vq_vae = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
        vq_vae2 = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
        vq_vae3 = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)

    model = VQVAEModel(encoder, decoder, encoder2, decoder2, encoder3, decoder3, vq_vae, vq_vae2, vq_vae3, pre_vq_conv1, pre_vq_conv2, pre_vq_conv3)
    checkpoint = tf.train.Checkpoint(module=model)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    if latest is not None:
        checkpoint.restore(latest)

    (winW, winH) = (res, res)
    mask = np.zeros((2048, 2048))

    print('Loading images from "%s"' % args.input_dir_test)
    image_filenames = glob.glob(args.input_dir_test)
    shape_c, shape_r = res, res

    for idx, path in enumerate(image_filenames):
        image = tiff.imread(path)
        name = os.path.splitext(os.path.basename(path))[0]

        # loop over the image pyramid
        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            img = np.expand_dims(np.array(window), 0)  * 2 / np.max(window) - 1
            model_results = model(tf.convert_to_tensor(img, dtype=tf.float32), is_training=False)

            err = tf.identity(tf.reduce_mean((model_results['x_recon'][0, :, :, 8] - img[0,:, :, 8]) ** 2))
            #err = tf.identity(model_results['recon_error']).numpy()
            # print(prediction)
            if err > thresh:
                mask[int(y + res / 2 - stepSize / 2):int(y + res / 2 + stepSize / 2),
                int(x + res / 2 - stepSize / 2):int(x + res / 2 + stepSize / 2)] = 255
        cv2.imwrite(f'{args.output_dir}detection_mask_{name}_32.png', mask)
        mask = np.zeros((2048, 2048))