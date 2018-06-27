#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import copy
import functools
import operator
import chainer
import chainer.links as L
import chainer.functions as F
from jsklearn.dnn.lstm import ConvLSTM2D


class LayerNormalization(chainer.Link):
    def __init__(self, size=None, eps=1e-6, initial_gamma=None,
                 initial_beta=None):
        super(LayerNormalization, self).__init__()
        if initial_gamma is None:
            initial_gamma = 1
        if initial_beta is None:
            initial_beta = 0

        with self.init_scope():
            self.gamma = chainer.variable.Parameter(initial_gamma)
            self.beta = chainer.variable.Parameter(initial_beta)
            self.eps = eps

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, size):
        self.gamma.initialize(size)
        self.beta.initialize(size)

    def __call__(self, x):
        if self.gamma.data is None:
            in_size = x[0].shape
            self._initialize_params(in_size)
        mean = F.broadcast_to(F.mean(x, keepdims=True), x.shape)
        var = F.broadcast_to(F.mean((x - mean) ** 2, keepdims=True), x.shape)
        mean = mean[0]
        var = var[0]
        return F.fixed_batch_normalization(
            x, self.gamma, self.beta, mean, var, self.eps)


class DEMEncoder(chainer.Chain):
    """Deep Episodic Memory Encoder"""

    def __init__(self, fc_channels=1000, fc_lstm_channels=1000, dropout_ratio=0.1,
                 norm_layer_cls=LayerNormalization):
        super(DEMEncoder, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=2, pad=2),
            conv_norm1=norm_layer_cls(),
            lstm1=ConvLSTM2D(32, 32, 5),
            lstm_norm1=norm_layer_cls(),
            #
            conv2=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm2=norm_layer_cls(),
            lstm2=ConvLSTM2D(32, 32, 5),
            lstm_norm2=norm_layer_cls(),
            #
            conv3=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm3=norm_layer_cls(),
            lstm3=ConvLSTM2D(32, 32, 3),
            lstm_norm3=norm_layer_cls(),
            #
            conv4=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            conv_norm4=norm_layer_cls(),
            lstm4=ConvLSTM2D(32, 64, 3),
            lstm_norm4=norm_layer_cls(),
            #
            conv5=L.Convolution2D(64, 64, 3, stride=2, pad=1),
            conv_norm5=norm_layer_cls(),
            lstm5=ConvLSTM2D(64, 64, 3),
            lstm_norm5=norm_layer_cls(),
            #
            fc_conv=L.Convolution2D(64, fc_channels, 4, stride=1, pad=0),
            fc_lstm=ConvLSTM2D(fc_channels, fc_lstm_channels, 1),
        )
        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(self.conv_norm1(h))
        #
        h = self.lstm1(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm1(h)
        #
        h = self.conv2(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm2(h))
        #
        h = self.lstm2(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm2(h)
        #
        h = self.conv3(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm3(h))
        #
        h = self.lstm3(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm3(h)
        #
        h = self.conv4(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm4(h))
        #
        h = self.lstm4(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm4(h)
        #
        h = self.conv5(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm5(h))
        #
        h = self.lstm5(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm5(h)
        #
        h = self.fc_conv(F.dropout(h, ratio=self.dropout_ratio))
        h = self.fc_lstm(F.dropout(h, ratio=self.dropout_ratio))
        #
        return F.concat((self.fc_lstm.c, self.fc_lstm.h))  # 2 * fc_lstm_channels


class DEMDecoder(chainer.Chain):
    def __init__(self, out_channels, fc_lstm_channels=1000, dropout_ratio=0.1,
                 norm_layer_cls=LayerNormalization):
        super(DEMDecoder, self).__init__(
            fc_lstm=ConvLSTM2D(fc_lstm_channels, fc_lstm_channels, 1, pad=0),
            fc_deconv=L.Deconvolution2D(fc_lstm_channels, 64, 4, stride=1, pad=0),
            #
            lstm1=ConvLSTM2D(64, 64, 3),
            lstm_norm1=norm_layer_cls(),
            deconv1=L.Deconvolution2D(64, 64, 3, stride=2, pad=1, outsize=(8, 8)),
            deconv_norm1=norm_layer_cls(),
            #
            lstm2=ConvLSTM2D(64, 64, 3),
            lstm_norm2=norm_layer_cls(),
            deconv2=L.Deconvolution2D(64, 64, 3, stride=2, pad=1, outsize=(16, 16)),
            deconv_norm2=norm_layer_cls(),
            #
            lstm3=ConvLSTM2D(64, 32, 3),
            lstm_norm3=norm_layer_cls(),
            deconv3=L.Deconvolution2D(32, 32, 5, stride=2, pad=2, outsize=(32, 32)),
            deconv_norm3=norm_layer_cls(),
            #
            lstm4=ConvLSTM2D(32, 32, 5),
            lstm_norm4=norm_layer_cls(),
            deconv4=L.Deconvolution2D(32, 32, 5, stride=2, pad=2, outsize=(64, 64)),
            deconv_norm4=norm_layer_cls(),
            #
            lstm5=ConvLSTM2D(32, 32, 5),
            lstm_norm5=norm_layer_cls(),
            deconv5=L.Deconvolution2D(32, out_channels, 5, stride=2, pad=2, outsize=(128, 128)),
        )

        # self.fc_lstm.copyparams(encoder.fc_lstm)
        # self.fc_lstm_h = copy.copy(encoder.fc_lstm.h)

        self.fc_lstm_channels = fc_lstm_channels
        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        assert isinstance(x, chainer.Variable)
        assert x.shape[1] == self.fc_lstm_channels * 2
        c, h = F.split_axis(x, 2, axis=1)
        self.fc_lstm.c, self.fc_lstm.h = c, h

        h = self.fc_lstm(h)
        h = self.fc_deconv(F.dropout(h, ratio=self.dropout_ratio))
        #
        h = self.lstm1(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm1(h)
        #
        h = self.deconv1(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm1(h))
        #
        h = self.lstm2(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm2(h)
        #
        h = self.deconv2(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm2(h))
        #
        h = self.lstm3(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm3(h)
        #
        h = self.deconv3(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm3(h))
        #
        h = self.lstm4(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm4(h)
        #
        h = self.deconv4(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm4(h))
        #
        h = self.lstm5(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm5(h)
        #
        o = self.deconv5(F.dropout(h, ratio=self.dropout_ratio))
        #
        return o


class DEMNet(chainer.Chain):
    """Composite model for Deep Episodic Memory Network"""

    def __init__(self, hidden_channels, out_channels, encoder_cls=None, decoder_cls=None, episode_size=None):
        super(DEMNet, self).__init__()

        if encoder_cls is None:
            encoder_cls = DEMEncoder
        if decoder_cls is None:
            decoder_cls = DEMDecoder

        with self.init_scope():
            self.encoder = encoder_cls(fc_lstm_channels=hidden_channels)
            self.decoder_reconst = decoder_cls(fc_lstm_channels=hidden_channels, out_channels=out_channels)
            self.decoder_pred = decoder_cls(fc_lstm_channels=hidden_channels, out_channels=out_channels)

        if episode_size is None:
            episode_size = 5

        self.episode_size = episode_size

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder_reconst.reset_state()
        self.decoder_pred.reset_state()

    def __call__(self, x):
        """x: (B, C, H, W)"""
        hidden = self.encoder(x)
        reconst = self.decoder_reconst(hidden)
        pred = self.decoder_pred(hidden)
        with chainer.cuda.get_device_from_id(self._device_id):
            pred_ret = chainer.Variable(pred.array.copy())
            reconst_ret = chainer.Variable(reconst.array.copy())
            hidden_ret = chainer.Variable(hidden.array.copy())
        return pred_ret, reconst_ret, hidden_ret
