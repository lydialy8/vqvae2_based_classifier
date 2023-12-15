import tensorflow as tf
import sonnet as snt

class ResidualStack(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = snt.Conv2D(
                output_channels=num_residual_hiddens,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="res3x3_%d" % i)
            conv1 = snt.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,dimension,
                 name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._dimension = dimension

        self._enc_1 = snt.Conv2D(
            output_channels=self._num_hiddens // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")
        self._enc_1a = snt.Conv2D(
            output_channels=self._num_hiddens//2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1a")
        self._enc_1b = snt.Conv2DTranspose(
            output_channels=self._num_hiddens,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1b")
        self._enc_1ab = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1ab")
        self._enc_2 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")
        self._enc_3 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        if self._dimension == 1:
            h = tf.nn.relu(self._enc_1a(x))
            h = tf.nn.relu(self._enc_1b(h))
        elif self._dimension == 2:
            h = tf.nn.relu(self._enc_1(x))
            h = tf.nn.relu(self._enc_2(h))
        else:
            h = tf.nn.relu(self._enc_1ab(x))
        return self._residual_stack(h)


class Decoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,dimension,channels,
                 name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self._dimension = dimension
        self.channels = channels
        self._dec_1 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = snt.Conv2DTranspose(
            output_channels=self._num_hiddens // 2,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")
        self._dec_3 = snt.Conv2DTranspose(
            output_channels=self.channels,
            output_shape=None,
            kernel_shape=(1, 1),
            stride=(1, 1),
            name="dec_3")
        self._dec_4 = snt.Conv2DTranspose(
            output_channels=self._num_hiddens,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_4")
        self._dec_5 = snt.Conv2DTranspose(
            output_channels=self.channels,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_5")
        self._dec_6 = snt.Conv2DTranspose(
            output_channels=self.channels,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        if self._dimension == 2:
            h = self._dec_4(h)
            x_recon = self._dec_5(h)
        elif self._dimension == 3:
            x_recon = self._dec_6(h)
        else:
            x_recon = self._dec_3(h)
        return x_recon


class VQVAEModel(snt.Module):
    def __init__(self, encoder, decoder
                 , encoder2, decoder2
                 , encoder3, decoder3
                 , vqvae, vqvae2,vqvae3,
                 pre_vq_conv1,
                 pre_vq_conv2,
                 pre_vq_conv3,
                 data_variance=0, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._encoder2 = encoder2
        self._decoder2 = decoder2
        self._encoder3 = encoder3
        self._decoder3 = decoder3
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._pre_vq_conv2 = pre_vq_conv2
        self._pre_vq_conv3 = pre_vq_conv3
        self._vqvae2 = vqvae2
        self._vqvae3 = vqvae3
        self._data_variance = data_variance
        self.upsample_t =snt.Conv2DTranspose(output_channels=64,output_shape=None,kernel_shape=(2, 2),stride=(2, 2),name="upsample_t")
        self.upsample_m =snt.Conv2DTranspose(output_channels=64,output_shape=None,kernel_shape=(4, 4),stride=(4, 4),name="upsample_m")


    def __call__(self, inputs, is_training):
        vq_output1, vq_output2, vq_output3, enc_b, enc_m, enc_t = self.encode(inputs, is_training)
        x_recon = self.decode(vq_output1['quantize'],vq_output2['quantize'],vq_output3['quantize'])
        recon_error = tf.reduce_mean((x_recon - inputs) ** 2) #/ self._data_variance
        loss = recon_error + vq_output1['loss'] + vq_output2['loss'] + vq_output3['loss']
        return {
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output1': vq_output1,
            'vq_output2': vq_output2,
            'vq_output3': vq_output3,
            'enc_b': enc_b,
            'enc_m': enc_m,
            'enc_t': enc_t
        }


    def encode(self, input, is_training):
        enc_b1 = self._encoder(input)
        enc_m1 = self._encoder2(enc_b1)
        enc_t = self._encoder3(enc_m1)

        z = self._pre_vq_conv1(enc_t)
        vq_output1 = self._vqvae(z, is_training=is_training)

        dec_t = self._decoder(vq_output1['quantize'])
        enc_m = tf.concat([dec_t, enc_m1], 3)

        z2 = self._pre_vq_conv2(enc_m)
        vq_output2 = self._vqvae2(z2, is_training=is_training)
        dec_m = self._decoder2(vq_output2['quantize'])
        enc_b = tf.concat([dec_m, enc_b1], 3)

        z3 = self._pre_vq_conv3(enc_b)
        vq_output3 = self._vqvae3(z3, is_training=is_training)

        return vq_output1, vq_output2, vq_output3, enc_b1, enc_m1, enc_t

    def decode(self, quant_t, quant_m, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant_tm = tf.concat([upsample_t, quant_m], 3)
        upsample_tm = self.upsample_m(quant_tm)
        quant = tf.concat([upsample_tm, quant_b], 3)
        dec = self._decoder3(quant)
        return dec