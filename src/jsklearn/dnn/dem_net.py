#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import copy
import chainer
import chainer.links as L
from jsklearn.dnn.lstm import ConvLSTM2D


class DEMEncoder(chainer.Chain):
    """Deep Episodic Memory Encoder"""

    def __init__(self, fc_channels=1000, fc_lstm_channels=1000, dropout_ratio=0.1):
        super(DEMEncoder, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=2),
            conv_norm1=L.LayerNormalization(),
            lstm1=ConvLSTM2D(32, 32, 5),
            lstm_norm1=L.LayerNormalization(),
            #
            conv2=L.Convolution2D(32, 32, 5, stride=2),
            conv_norm2=L.LayerNormalization(),
            lstm2=ConvLSTM2D(32, 32, 5),
            lstm_norm2=L.LayerNormalization(),
            #
            conv3=L.Convolution2D(32, 32, 5, stride=2),
            conv_norm3=L.LayerNormalization(),
            lstm3=ConvLSTM2D(32, 32, 3),
            lstm_norm3=L.LayerNormalization(),
            #
            conv4=L.Convolution2D(32, 32, 3, stride=2),
            conv_norm4=L.LayerNormalization(),
            lstm4=ConvLSTM2D(32, 64, 3),
            lstm_norm4=L.LayerNormalization(),
            #
            conv5=L.Convolution2D(64, 64, 3, stride=2),
            conv_norm5=L.LayerNormalization(),
            lstm5=ConvLSTM2D(64, 64, 3),
            lstm_norm5=L.LayerNormalization(),
            #
            fc_conv=L.Convolution2D(64, fc_channels, 3, stride=2),
            fc_lstm=ConvLSTM2D(fc_channels, fc_lstm_channels, 1),
        )
        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if hasattr(link, "reset_state"):
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
        return self.fc_lstm.c


class DEMDecoder(chainer.Chain):
    def __init__(self, encoder, out_channels, fc_lstm_channels=1000, dropout_ratio=0.1):
        super(DEMDecoder, self).__init__(
            fc_lstm=ConvLSTM(fc_lstm_channels, fc_lstm_channels, 1),
            fc_deconv=L.Deconvolution2D(fc_lstm_channels, 64, 4, stride=1),
            #
            lstm1=ConvLSTM(64, 64, 3),
            lstm_norm1=L.LayerNormalization(),
            deconv1=L.Deconvolution2D(64, 64, 3, stride=2),
            deconv_norm1=L.LayerNormalization(),
            #
            lstm2=ConvLSTM(64, 64, 3),
            lstm_norm2=L.LayerNormalization(),
            deconv2=L.Deconvolution2D(64, 64, 3, stride=2),
            deconv_norm2=L.LayerNormalization(),
            #
            lstm3=ConvLSTM(64, 32, 3),
            lstm_norm3=L.LayerNormalization(),
            deconv3=L.Deconvolution2D(32, 32, 5, stride=2),
            deconv_norm3=L.LayerNormalization(),
            #
            lstm4=ConvLSTM(32, 32, 5),
            lstm_norm4=L.LayerNormalization(),
            deconv4=L.Deconvolution2D(32, 32, 5, stride=2),
            deconv_norm4=L.LayerNormalization(),
            #
            lstm5=ConvLSTM(32, 32, 5),
            lstm_norm5=L.LayerNormalization(),
            deconv5=L.Deconvolution2D(32, out_channels, 5, stride=2),
        )

        self.fc_lstm.copyparams(encoder.fc_lstm)
        self.fc_lstm_h = copy.copy(encoder.fc_lstm.h)

        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if hasattr(link, "reset_state"):
                link.reset_state()



    def __call__(self, x=None):
        if x is None:
            x = copy.copy(self.fc_lstm_h)

        h = self.fc_lstm(x)
        h = self.fc_deconv(F.dropout(h), ratio=self.dropout_ratio)
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
