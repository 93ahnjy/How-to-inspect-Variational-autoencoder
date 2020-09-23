import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr



def get_encoder_layer(model, name):
    print(dict(model.encoder.named_children()).keys())
    t = list(dict(model.encoder.named_children()).keys()).index(name)
    return  model.encoder[t]


def get_decoder_layer(model, name):
    return list(dict(model.decoder.named_children()).keys()).index(name)



def get_discriminator_layer(model, name):
    print(dict(model.named_children()).keys())
    t = list(dict(model.named_children()).keys()).index(name)
    return model[t]







def minmax_standardization(input_data):
    min_, max_ = input_data.min(), input_data.max()
    return (input_data - min_).div(max_ - min_ + 1e-8).data  # normalize


def mean_normalization(input_data):
    mean_, std_ = input_data.mean(), input_data.std()
    return (input_data - mean_) / std_





class Print_layer(nn.Module):
    def __init__(self, string=None):
        super(Print_layer, self).__init__()
        self.string = string

    def forward(self, x):
        print(x.size())
        if self.string is not None:
            print(self.string)
        return x




class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)




class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)








class Reparam_block(nn.Module):
    def __init__(self, in_channel, latent_size):
        super(Reparam_block, self).__init__()

        self.mu_fc = nn.Linear(in_channel, latent_size)
        self.mu_bn = nn.BatchNorm1d(latent_size)
        self.mu_relu = nn.ReLU(inplace=True)


        self.std_fc = nn.Linear(in_channel, latent_size)
        self.std_bn = nn.BatchNorm1d(latent_size)
        self.std_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        #mu     = self.mu_relu(self.mu_bn(self.mu_fc(x)))
        #logvar = self.std_relu(self.std_bn(self.std_fc(x)))


        mu     = self.mu_fc(x)
        logvar = self.std_fc(x)

        return mu, logvar






class Reparam_block_spatial(nn.Module):
    def __init__(self, in_channel, latent_size):
        super(Reparam_block_spatial, self).__init__()

        self.mu_conv = nn.Conv2d(in_channel, latent_size, 3, stride=1, padding=1, bias=False)
        self.mu_bn   = nn.BatchNorm2d(latent_size)
        self.mu_relu = nn.ReLU(inplace=True)


        self.std_conv = nn.Conv2d(in_channel, latent_size, 3, stride=1, padding=1, bias=False)
        self.std_bn   = nn.BatchNorm2d(latent_size)
        self.std_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        mu     = self.mu_relu(self.mu_bn(self.mu_conv(x)))
        logvar = self.std_relu(self.std_bn(self.std_conv(x)))
        return mu, logvar










def create_encoder(hidden_channels, kernels, strides, inout_channel, use_resnet18_pretrained):

    modules_E = []
    h_c = hidden_channels
    k = kernels
    c_in = inout_channel



    if use_resnet18_pretrained:
        encoder = models.resnet18(pretrained=True)
        encoder.fc = nn.Linear(512, h_c[-1])
        return encoder



    else:

        for i in range(len(h_c)):

            if i == 0:
                modules_E.append(('conv2d_{}'.format(str(i)),nn.Conv2d(c_in, h_c[0], k[0], stride=strides[0], padding=1, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[0])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))


            elif i < len(h_c) - 1:
                modules_E.append(('conv2d_{}'.format(str(i)), nn.Conv2d(h_c[i - 1], h_c[i], k[i], stride=strides[i], padding=1, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[i])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))

            elif i == len(h_c) - 1:
                modules_E.append(('conv2d_{}'.format(str(i)), nn.Conv2d(h_c[i - 1], h_c[i], k[i], stride=strides[i], padding=0, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[i])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))

        return nn.Sequential(OrderedDict(modules_E))

















def create_decoder(hidden_channels, kernels, strides, inout_channel):

    modules_D = []
    h_c = hidden_channels
    k = kernels
    c_out = inout_channel


    for i in reversed(range(len(h_c))):

        l_idx = len(h_c) - i - 1

        if i == len(h_c) - 1:
            #modules_D.append(('print_{}'.format(str(i)), Print_layer()))

            modules_D.append(('conv2d_T_{}'.format(str(l_idx)), nn.ConvTranspose2d(h_c[i], h_c[i - 1], k[i], stride=strides[i], padding=0,bias=False)))
            modules_D.append(('bn_{}'.format(str(l_idx)), nn.BatchNorm2d(h_c[i - 1])))
            modules_D.append(('relu_{}'.format(str(l_idx)), nn.LeakyReLU(0.2, inplace=True)))
            #modules_D.append(('print_{}'.format(str(i)), Print_layer()))

        elif 0 < i < len(h_c) - 1:
            modules_D.append(('conv2d_T_{}'.format(str(l_idx)), nn.ConvTranspose2d(h_c[i], h_c[i - 1], k[i], stride=strides[i], padding=1,bias=False)))
            modules_D.append(('bn_{}'.format(str(l_idx)), nn.BatchNorm2d(h_c[i - 1])))
            modules_D.append(('relu_{}'.format(str(l_idx)), nn.LeakyReLU(0.2, inplace=True)))
            #modules_D.append(('print_{}'.format(str(i)), Print_layer()))

        elif i == 0:
            modules_D.append(('conv2d_T_{}'.format(str(l_idx)), nn.ConvTranspose2d(h_c[0], c_out, k[0], stride=strides[0], padding=1, bias=False)))
            #modules_D.append(('bn_{}'.format(str(l_idx)), nn.BatchNorm2d(c_out)))
            modules_D.append(('Tanh_{}'.format(str(l_idx)), nn.Tanh()))
            #modules_D.append(('print_{}'.format(str(i)), Print_layer()))

    return nn.Sequential(OrderedDict(modules_D))















def create_discriminator(hidden_channels, kernels, strides, inout_channel, use_resnet18_pretrained):

    modules_E = []
    h_c = hidden_channels
    k = kernels
    c_in = inout_channel



    if use_resnet18_pretrained:
        encoder = models.resnet18(pretrained=True)
        encoder.fc = nn.Linear(512, 1)
        return encoder



    else:

        for i in range(len(h_c)):

            if i == 0:
                modules_E.append(('conv2d_{}'.format(str(i)),nn.Conv2d(c_in, h_c[0], k[0], stride=strides[0], padding=1, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[0])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))


            elif i < len(h_c) - 1:
                modules_E.append(('conv2d_{}'.format(str(i)), nn.Conv2d(h_c[i - 1], h_c[i], k[i], stride=strides[i], padding=1, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[i])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))

            elif i == len(h_c) - 1:
                modules_E.append(('conv2d_{}'.format(str(i)), nn.Conv2d(h_c[i - 1], h_c[i], k[i], stride=strides[i], padding=0, bias=False)))
                modules_E.append(('bn_{}'.format(str(i)), nn.BatchNorm2d(h_c[i])))
                modules_E.append(('relu_{}'.format(str(i)), nn.ReLU(inplace=True)))
                modules_E.append(('Flatten_{}'.format(str(i)), Flatten()))
                modules_E.append(('FC_{}'.format(str(i)), nn.Linear(h_c[i], 1)))
                #modules_E.append(('print_{}'.format(str(i)), Print_layer()))


        return nn.Sequential(OrderedDict(modules_E))
























'''

class Arcfacelayer(Layer):
    # s:softmaxの温度パラメータ, m:margin
    def __init__(self, output_dim, s=30, m=0.50, easy_margin=False):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        super(Arcfacelayer, self).__init__()

    # 重みの作成
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)


    # mainの処理 
    def call(self, x):
        y = x[1]
        x_normalize = tf.math.l2_normalize(x[0]) # x = x'/ ||x'||2
        k_normalize = tf.math.l2_normalize(self.kernel) # Wj = Wj' / ||Wj'||2

        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m

        cosine = K.dot(x_normalize, k_normalize) # W.Txの内積
        sine = K.sqrt(1.0 - K.square(cosine))

        phi = cosine * cos_m - sine * sin_m #cos(θ+m)の加法定理

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine) 

        else:
            phi = tf.where(cosine > th, phi, cosine - mm) 


        # 正解クラス:cos(θ+m) 他のクラス:cosθ 
        output = (y * phi) + ((1.0 - y) * cosine) 
        output *= self.s

        return output


'''















class VAE_attention(object):

    def __init__(self, arch, target_layer):

        self.model_arch = arch
        self.attention_maps = []
        self.att_mean = []
        self.att_std = []

        self.gradients = dict()
        self.activations = dict()

        self.model_arch.is_training = False

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)





    def forward(self, input, retain_graph=False):  # input should be (1, c, h, w) shape

        b, c, h, w = input.size()

        output, latent, _, _ = self.model_arch(input)

        res = torch.abs(input  - output + 1e-4).max(dim=1, keepdim=True)[0]
        res = F.interpolate(res, size=self.activations['value'].size()[2])


        if latent.size().__len__() > 2:
            lb, lc, lh, lw = latent.size()
            latent = F.avg_pool2d(latent, lh)

        latent = latent.squeeze()


        print("Evaluating")
        print(latent.view(-1))
        print(latent.view(-1).mean().data, latent.view(-1).std().data)

        self.model_arch.zero_grad()

        for i in range(len(latent)):

            latent[i].backward(retain_graph=retain_graph)

            gradients, activations = self.gradients['value'], self.activations['value']
            b, k, u, v = gradients.size()

            print(b,k,u,v)

            alpha = nn.AdaptiveAvgPool2d(1)(gradients)
            weights = alpha.view(b, k, 1, 1)


            attention_map = (weights * activations).sum(1, keepdim=True)  # get gradcam & resize
            attention_map = F.relu(attention_map)
            attention_map = F.upsample(attention_map, size=(h, w), mode='bilinear', align_corners=True)

            #attention_map = minmax_standardization(attention_map)  # standardization

            self.attention_maps.append(attention_map)


        attention_map = torch.cat(self.attention_maps, dim=0)
        self.attention_maps = []                                   # Without this, memory will not release.

        return output, attention_map



    def __call__(self, input, retain_graph=False):
        return self.forward(input, retain_graph=True)






























class Dis_attention(object):

    def __init__(self, arch, target_layer):

        self.model_arch = arch
        self.attention_maps = []
        self.att_mean = []
        self.att_std = []

        self.gradients = dict()
        self.activations = dict()

        self.model_arch.is_training = False

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)





    def forward(self, input, retain_graph=False):  # input should be (1, c, h, w) shape

        b, c, h, w = input.size()

        logit = self.model_arch(input)



        print(logit.size())


        self.model_arch.zero_grad()
        print(logit.size())

        logit.backward(retain_graph=retain_graph)

        gradients, activations = self.gradients['value'], self.activations['value']
        b, k, u, v = gradients.size()


        alpha = nn.AdaptiveAvgPool2d(1)(gradients)
        weights = alpha.view(b, k, 1, 1)

        attention_map = (weights * activations ).sum(1, keepdim=True)  # get gradcam & resize
        attention_map = F.relu(attention_map)
        attention_map = F.upsample(attention_map, size=(h, w), mode='bilinear', align_corners=True)


        #attention_map = minmax_standardization(attention_map)  # standardization


        return attention_map

    def __call__(self, input, retain_graph=False):
        return self.forward(input, retain_graph=True)





