import numpy as np
import sonnet as snt
from model import Encoder, Decoder, VQVAEModel
from utils import *
import os
import pathlib
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from create_calibration_datasets import computeNDVIMatrix
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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


def reconstruct(model, val_dataset, output_dir, val_dataset_gt):

    pathlib.Path(os.path.join(output_dir, 'mse')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_dir, 'recon')).mkdir(parents=True, exist_ok=True)
    for step_index, data in enumerate(val_dataset):
        model_results = model(data[0], is_training=False)
        model_output = (tf.identity(model_results['x_recon']) + 1) * 255 / 2
        input_image = (tf.identity(data[0]) + 1) * 255 / 2
        #input_image = np.stack((input_image.numpy()[0, :, :, 3], input_image.numpy()[0, :, :, 2], input_image.numpy()[0, :, :, 1]),axis=2).astype(np.uint8)
        #recon_img = np.stack((model_output.numpy()[0, :, :, 3], model_output.numpy()[0, :, :, 2], model_output.numpy()[0, :, :, 1]),axis=2).astype(np.uint8)
        fig_img = plt.figure()
        ax1 = fig_img.add_subplot(1, 2, 2)
        plt.axis('off')
        ax1.imshow(model_output[0], label='reconstructed samples', cmap='gray')
        ax2 = fig_img.add_subplot(1, 2, 1)
        ax2.imshow(input_image[0], label='input samples', cmap='gray')
        ax1.title.set_text('reconstructed image')
        ax2.title.set_text('GAN image')
        plt.axis('off')
        fig_img.savefig(os.path.join(output_dir, 'recon','reconstruction_fake_%06d.png' % (step_index + 1)))

        for i in range(0, 1):
            plt.figure(figsize=(12, 4))
            err1= tf.identity((model_results['x_recon'][0, :, :, i] - data[0][0, :, :, i]) ** 2).numpy()
            plt.imshow(err1,cmap='Reds', interpolation='nearest', label='real samples' + str(i))
            plt.savefig(os.path.join(output_dir, 'mse','bands_mse_fake_%06d_%06d.png' % (step_index, i + 1)))
    for step_index, data in enumerate(val_dataset_gt):
        model_results = model(data[0], is_training=False)
        model_output = (tf.identity(model_results['x_recon']) + 1) * 255 / 2
        input_image = (tf.identity(data[0]) + 1) * 255 / 2
        fig_img = plt.figure()
        ax1 = fig_img.add_subplot(1, 2, 2)
        plt.axis('off')
        ax1.imshow(model_output[0], label='reconstructed samples', cmap='gray')
        ax2 = fig_img.add_subplot(1, 2, 1)
        ax2.imshow(input_image[0], label='input samples', cmap='gray')
        ax1.title.set_text('reconstructed image')
        ax2.title.set_text('input image')
        plt.axis('off')
        fig_img.savefig(os.path.join(output_dir, 'recon','reconstruction_gt_%06d.png' % (step_index + 1)))
        for i in range(0, 1):
            plt.figure(figsize=(12, 4))
            err1= tf.identity((model_results['x_recon'][0, :, :, i] - data[0][0, :, :, i]) ** 2).numpy()
            plt.imshow(err1,cmap='Reds', interpolation='nearest', label='real samples' + str(i))
            plt.savefig(os.path.join(output_dir, 'mse','bands_mse_real_%06d_%06d.png' % (step_index, i + 1)))

def getReconLists(dataset, model, channels_nb):
    recon_error_list = []
    recon_error_list_bands = []
    class_list = []
    filename_list = []
    for step_index, data in enumerate(dataset):
        img = data[0]
        src_img = (data[1] + 1 )/2
        model_results = model(img, is_training=False)
        #snowflag, vegetationflag, barrenflag, waterflag = computeNDVIMatrix(src_img.numpy()[0, :, :, 2], src_img.numpy()[0, :, :, 3], src_img.numpy()[0, :, :, 7], src_img.numpy()[0, :, :, 11], src_img.numpy()[0, :, :, 1])
        """snowflag, vegetationflag, barrenflag, waterflag = computeNDVIMatrix(src_img.numpy()[0, :, :, 1],
                                                                            src_img.numpy()[0, :, :, 2],
                                                                            src_img.numpy()[0, :, :, 3],
                                                                            src_img.numpy()[0, :, :, 0],
                                                                            src_img.numpy()[0, :, :, 0])

        if snowflag:
           img_class = 0
        elif barrenflag:
           img_class = 2
        elif vegetationflag or waterflag:
           img_class = 1
        else:
            img_class = 3
        class_list.append(img_class)"""
        filename_list.append(str(data[2].numpy()[0]))
        recon_error_list.append(tf.identity(model_results['recon_error']).numpy())
        err = []
        for i in range(0,channels_nb):
            err.append(tf.identity(tf.reduce_mean((model_results['x_recon'][0,:,:,i] - img[0,:,:,i]) ** 2)))
        recon_error_list_bands.append(err)
    return recon_error_list, recon_error_list_bands,class_list,filename_list

def getHistogram(recon_error_list_real, recon_error_list_gen, output, i):
    # histogram
    plt.figure()
    plt.hist(recon_error_list_real, bins=300, alpha=0.5, color="green", lw=0, label='real samples')
    plt.hist(recon_error_list_gen, bins=300, alpha=0.5, color="red", lw=0, label='generated samples' )
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("count of images")
    plt.legend()
    plt.savefig(os.path.join(output, 'hist', 'hist_bands_%s.png' % str(i)))

def savePkls(recon_error_list_real,recon_error_list_gen, class_list_real, class_list_gan, output, i,filename1, filename2):
    labels = [1] * len(recon_error_list_real) + [0] * len(recon_error_list_gen)
    metric = np.concatenate((recon_error_list_real,recon_error_list_gen))
    img_class = np.concatenate((class_list_real,class_list_gan))
    filenames = np.concatenate((filename1,filename2))
    with open(os.path.join(output, 'pkls', "labels_%s.pkl" % str(i)), "wb") as f:
        pickle.dump(labels, f)
    with open(os.path.join(output, 'pkls', "metrics_%s.pkl" % str(i)), "wb") as f:
        pickle.dump(metric, f)
    with open(os.path.join(output, 'pkls', "class_%s.pkl" % str(i)), "wb") as f:
        pickle.dump(img_class, f)
    with open(os.path.join(output, 'pkls', "filenames_%s.pkl" % str(i)), "wb") as f:
        pickle.dump(filenames, f)
    return labels, metric

def getROC(labels,metric,output,i):
    fpr, tpr, thresholds = metrics.roc_curve(labels, metric, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output, 'plots', 'roc_%s.png' % str(i)))

def test(model, val_dataset,val_dataset_gen, output, channels_nb):
    pathlib.Path(os.path.join(output, 'hist')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, 'plots')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, 'pkls')).mkdir(parents=True, exist_ok=True)
    recon_error_list_real, recon_error_list_realbands, class_list_real,filename_list_real = getReconLists(val_dataset, model, channels_nb)
    recon_error_list_gen, recon_error_list_genbbands, class_list_gan,filename_list_gan = getReconLists(val_dataset_gen, model, channels_nb)
    labels, metric = savePkls(recon_error_list_real, recon_error_list_gen, class_list_real, class_list_gan, output, 'all',filename_list_real,filename_list_gan)
    getHistogram(recon_error_list_real, recon_error_list_gen, output, 'all')
    getROC(labels, metric, output, 'a')

    """for i in range(0,channels_nb):
        labels, metric = savePkls(tf.identity(recon_error_list_realbands).numpy()[:, i], tf.identity(recon_error_list_genbbands).numpy()[:, i], output, i)
        getHistogram(tf.identity(recon_error_list_realbands).numpy()[:, i], tf.identity(recon_error_list_genbbands).numpy()[:, i], output, i)
        getROC(labels, metric, output, i)"""

def test(model,val_dataset_gen, output, channels_nb):
    pathlib.Path(os.path.join(output, 'hist')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, 'plots')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output, 'pkls')).mkdir(parents=True, exist_ok=True)
    recon_error_list_real,  class_list_real, filename_list_real= [], [], []
    recon_error_list_gen, recon_error_list_genbbands, class_list_gan,filename_list_gan = getReconLists(val_dataset_gen, model, channels_nb)
    labels, metric = savePkls(recon_error_list_real, recon_error_list_gen, class_list_real, class_list_gan, output, 'all',filename_list_real,filename_list_gan)
    #getHistogram(recon_error_list_real, recon_error_list_gen, output, 'all')
    #getROC(labels, metric, output, 'a')

    """for i in range(0,channels_nb):
        labels, metric = savePkls(tf.identity(recon_error_list_realbands).numpy()[:, i], tf.identity(recon_error_list_genbbands).numpy()[:, i], output, i)
        getHistogram(tf.identity(recon_error_list_realbands).numpy()[:, i], tf.identity(recon_error_list_genbbands).numpy()[:, i], output, i)
        getROC(labels, metric, output, i)"""

def train(model,optimizer, train_dataset):
    #@tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            model_output = model(data[0], is_training=True)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(model_output['loss'], trainable_variables)
        optimizer.apply(grads, trainable_variables)

        return model_output

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    for step_index, data in enumerate(train_dataset):
        train_results = train_step(data)
        train_losses.append(train_results['loss'])
        train_recon_errors.append(train_results['recon_error'])
        train_perplexities.append(train_results['vq_output3']['perplexity'])
        train_vqvae_loss.append(train_results['vq_output3']['loss'])
        train_perplexities.append(train_results['vq_output2']['perplexity'])
        train_vqvae_loss.append(train_results['vq_output2']['loss'])
        train_perplexities.append(train_results['vq_output1']['perplexity'])
        train_vqvae_loss.append(train_results['vq_output1']['loss'])

        if (step_index + 1) % 100 == 0:
            print('%d train loss: %f ' % (step_index + 1,
                                          np.mean(train_losses[-100:])) +
                  ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                  ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                  ('vqvae loss : %.3f' % np.mean(train_vqvae_loss[-100:])))
        if step_index and not step_index % args.save_steps:
            checkpoint.save(os.path.join(checkpoint_dir, 'network_snapshot_%06d' % (step_index)))
        if step_index == args.num_training_updates:
            checkpoint.save(os.path.join(checkpoint_dir, 'network_snapshot_final'))
            break
    return train_losses, train_recon_errors, train_perplexities, train_vqvae_loss

def vq_model(args):
    # Data Loading.
    # Set hyper-parameters.
    num_hiddens = args.hiddens
    num_residual_hiddens = args.residual_hiddens
    num_residual_layers = args.residual_layers
    embedding_dim = args.embedding_dim
    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = args.num_embeddings
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

    # # Build modules.
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,1)
    encoder2 = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,2)
    encoder3 = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens,3)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,3,args.channels)
    decoder2 = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,2,args.channels)
    decoder3 = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens,1,args.channels)

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

    model = VQVAEModel(encoder, decoder, encoder2, decoder2, encoder3, decoder3, vq_vae, vq_vae2, vq_vae3, pre_vq_conv1, pre_vq_conv2, pre_vq_conv3, embedding_dim)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--channels', type=int, default=1
                        , choices=[3, 4, 13],help='number of channels for the model')
    parser.add_argument('--pristine_data_channels', type=int, default=4
                        , choices=[4, 13],help='number of channels in an image')
    parser.add_argument('--gan_data_channels', type=int, default=3
                        , choices=[4, 13],help='number of channels in an image')
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
    parser.add_argument('--mode_4on13', type=str2bool, default=False, help='test 4 bands on 13 bands')
    parser.add_argument('--vq_use_ema', type=str2bool, default=True, help='uses EMA optimizer instead of Adam for codebook updates')
    parser.add_argument('--training_flag', type=str2bool, default=False, help='train or just test')
    parser.add_argument('--reconstruct_images', type=str2bool, default=False, help='save reconstructed images and their relative squared error')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/checkptperband/B0',  help='Directory for saving checkpoints')
    parser.add_argument('--input_dir_train', type=str, default='D:\\detection_experiments\\datasets\\australia_pristine\\val\\real\\*',  help='Directory input training images')
    parser.add_argument('--input_dir_test', type=str, default='D:\\detection_experiments\\datasets\\scand_4bands\\train\\*\\real*\\*',  help='Directory input test images')
    parser.add_argument('--input_dir_gen_test', type=str, default='C:\\Users\\lydia\\OneDrive\\Desktop\\doesntexit\\*', help='Directory input test images')#D:\\detection_experiments\\datasets\\scand_4bands\\train\\*\\fake*\\*

    #parser.add_argument('--input_dir_test', type=str, default='D:\\detection_experiments\\datasets\\calibration_datasets\\vegetation_13\\real\\*',  help='Directory input test images')
    #parser.add_argument('--input_dir_gen_test', type=str, default='D:\\detection_experiments\\datasets\\calibration_datasets\\vegetation_13\\fake\\*',  help='Directory input test images')
    parser.add_argument('--output_dir', type=str, default='./results_21_04_2022/doesntexit_train/B0', help='Directory for saving generated images')
    parser.add_argument('--band', type=int, default=1,help='band')
    parser.add_argument('--test_onlypristine', type=str2bool, default=True, help=' one class only')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    learning_rate = args.lr
    training_flag = args.training_flag
    checkpoint_dir = args.checkpoint_dir
    model = vq_model(args)
    checkpoint = tf.train.Checkpoint(module=model)
    latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    if latest is not None:
        checkpoint.restore(latest)
    if training_flag:
        optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
        train_dataset = data_generator(args.input_dir_train, args.batch_size, args.channels, args.band, args.image_size, args.pristine_data_channels)
        train_losses, train_recon_errors, train_perplexities, train_vqvae_loss = train(model, optimizer, train_dataset)
        plot_loss(train_recon_errors, train_perplexities)
    elif not args.reconstruct_images:
        #val_dataset = data_generator_val(args.input_dir_test, args.batch_size, args.channels, args.band, args.image_size, args.pristine_data_channels)
        val_dataset_gen = data_generator_val(args.input_dir_gen_test, args.batch_size, args.channels, args.band, args.image_size, args.gan_data_channels)
        test(model, val_dataset_gen, args.output_dir, args.channels)
    if args.reconstruct_images:
        val_dataset = data_generator_val(args.input_dir_test, args.batch_size, args.channels, args.band, args.image_size, args.pristine_data_channels, args.mode_4on13)
        val_dataset_gen = data_generator_val(args.input_dir_gen_test, args.batch_size, args.channels, args.band, args.image_size, args.gan_data_channels, args.mode_4on13)
        reconstruct(model, val_dataset_gen, args.output_dir,val_dataset)