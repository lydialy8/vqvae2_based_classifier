import tifffile as tiff
import numpy as np
import glob
from utils import *
from vq_vae import vq_model
from sklearn.ensemble import IsolationForest
import cv2

height = width = res = 512
stepSize = 512
shape = height, width

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def getReconLists(img_bnd, model):
    img_bnd = np.expand_dims(img_bnd,3)
    img_bnd = tf.convert_to_tensor(img_bnd, dtype=tf.float32)
    model_results = model(img_bnd, is_training=False)
    err = tf.identity((model_results['x_recon'][0, :, :, 0] - img_bnd[0, :, :, 0]) ** 2).numpy()
    return err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--channels', type=int, default=1
                        , choices=[3, 4, 13],help='number of channels for the model')
    parser.add_argument('--data_channels', type=int, default=13
                        , choices=[4, 13],help='number of channels in an image')
    parser.add_argument('--hiddens', type=int, default=256, help='number of hiddens')
    parser.add_argument('--residual_hiddens', type=int, default=64, help='number of residual hidden layers')
    parser.add_argument('--residual_layers', type=int, default=2, help='number of residual layers')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding code dimensions')
    parser.add_argument('--num_embeddings', type=int, default=512, help='capacity of embeddings')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost')
    parser.add_argument('--decay', type=float, default=0.99, help='decay of learning rate')
    parser.add_argument('--vq_use_ema', type=str2bool, default=True, help='uses EMA optimizer instead of Adam for codebook updates')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/checkptperband/',  help='Directory for saving checkpoints')
    parser.add_argument('--input_dir_test', type=str, default='D:\\styleTransferExperiments\\repositories\\pix2pix_cyclegan_multispectral\\results_09-03_spliced2\\ex2\\spliced',  help='Directory input test images')
    parser.add_argument('--output_dir', type=str, default='D:\\styleTransferExperiments\\repositories\\pix2pix_cyclegan_multispectral\\results_09-03_spliced2\\localization_heatmap\\', help='Directory for saving generated images')


    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = vq_model(args)
    checkpoint = tf.train.Checkpoint(module=model)
    (winW, winH) = (res, res)
    mask = np.ones((2048, 2048))
    print('Loading images from "%s"' % args.input_dir_test)
    image_filenames = glob.glob(os.path.join(args.input_dir_test, '*.tif'))
    shape_c, shape_r = res, res
    for idx, path in enumerate(image_filenames):
        image = tiff.imread(path)
        name = os.path.splitext(os.path.basename(path))[0]
        # loop over the image pyramid
        overall_heatmap = np.zeros((args.data_channels, 2048, 2048))
        for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            for i in range(0,args.data_channels):
                latest = tf.train.latest_checkpoint(os.path.join(args.checkpoint_dir,f'B{i}'))
                if latest is not None:
                    checkpoint.restore(latest)
                img_bnd = np.expand_dims(np.array(window[:, :, i]), 0) / np.max(window[:, :, i])
                overall_heatmap[i,int(y + res / 2 - stepSize / 2):int(y + res / 2 + stepSize / 2), int(x + res / 2 - stepSize / 2):int(x + res / 2 + stepSize / 2)] = getReconLists(img_bnd, model)
        data = np.transpose(overall_heatmap.reshape((13, 2048 * 2048)),(1,0))
        clf_if = IsolationForest(random_state=0).fit_predict(data)
        #plt.figure(figsize=(12, 4))
        #plt.imshow(clf_if.reshape((2048,2048)), cmap='Reds', interpolation='nearest')

        cv2.imwrite(f'{args.output_dir}detection_mask_{name}.png',np.where(clf_if.reshape((2048,2048))*255 ==255, 0, 255) )
        mask = np.ones((2048, 2048))














