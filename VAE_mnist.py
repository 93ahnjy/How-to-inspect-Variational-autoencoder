# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import Print_layer

from Dataset import load_data
from plot import *
from loss_functions import MADLoss

from torch.autograd import Variable
from torchvision.utils import save_image












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
            self.gradients['value'] = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations['value'] = output.detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)





    def forward(self, input, retain_graph=False):  # input should be (1, c, h, w) shape

        b, c, h, w = input.size()

        # recon_batch, mu, log_var = self.model(data)
        output, latent, mu, logvar = self.model_arch(input)
        latent = latent.squeeze()


        #print("Evaluating")
        #print(mu.view(-1))
        #print(latent.view(-1).mean().data, latent.view(-1).std().data)

        self.model_arch.zero_grad()

        for i in range(len(latent)):


            latent[i].backward(retain_graph=retain_graph)

            gradients, activations = self.gradients['value'], self.activations['value']
            b, k, u, v = gradients.size()

            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)


            attention_map = torch.mul(weights,activations).sum(1, keepdim=True)  # get gradcam & resize
            attention_map = F.relu(attention_map)
            attention_map = F.upsample(attention_map, size=(h, w), mode='bilinear', align_corners=True)

            attention_map = minmax_standardization(attention_map)  # standardization

            #if attention_norm.mean() < 0.3:
            self.attention_maps.append(attention_map)




        attention_map = torch.cat(self.attention_maps, dim=0)
        self.attention_maps = []                                   # Without this, memory will not release.

        return output, attention_map



    def __call__(self, input, retain_graph=False):
        return self.forward(input, retain_graph=True)








class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 7, 7)




'''
torch.Size([128, 3, 32, 32])
torch.Size([128, 32, 15, 15])
torch.Size([128, 64, 6, 6])
torch.Size([128, 128, 2, 2])
torch.Size([128, 512])
torch.Size([128, 512])
torch.Size([128, 512, 1, 1])
torch.Size([128, 128, 3, 3])
torch.Size([128, 64, 7, 7])
torch.Size([128, 32, 15, 15])
torch.Size([128, 3, 32, 32])
'''


class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()



        self.encoder = nn.Sequential(
            #Print_layer(),

            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #Print_layer(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #Print_layer(),

            Flatten(),
            #Print_layer(),

            nn.Linear(128 * 7 * 7, h_dim),
            #nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            #Print_layer(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)


        self.decoder = nn.Sequential(
            #Print_layer(),
            nn.Linear(z_dim, h_dim),
            #nn.BatchNorm1d(h_dim),
            nn.ReLU(),

            nn.Linear(h_dim, 128 * 7 * 7),
            #nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
            #Print_layer(),

            UnFlatten(),
            nn.ReLU(),
            #Print_layer(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #Print_layer(),

            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            #Print_layer(),
        )


    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()

        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z_, mu, logvar = self.bottleneck(h)

        return self.decoder(z_), z_, mu, logvar


    def decode(self, z_):
        return self.decoder(z_)













class Model():
    def __init__(self, settings):
        super(Model, self).__init__()

        # build model
        self.batch_size = settings['batch_size']
        self.h_dim = settings['h_dim']
        self.z_dim = settings['z_dim']
        self.in_channel = settings['image_shape'][2]
        self.input_size = settings['image_shape'][:2]
        self.model = VAE(image_channels=self.in_channel, h_dim=self.h_dim, z_dim=self.z_dim)

        self.train_dir = settings['train_dir']
        self.test_dir = settings['test_dir']

        self.train_data_loader = load_data('mnist', self.batch_size, self.input_size,
                                      data_dir=self.train_dir,
                                      mode='train',
                                      color_channel=self.in_channel)

        self.test_data_loader = load_data('mnist', 64, self.input_size,
                                     data_dir=self.test_dir,
                                     mode='test',
                                     color_channel=self.in_channel)

        self.batch_size = 128
        self.input_size = settings['image_shape'][:2]


        self.save_model_name = './models/VAE_mnist_anomaly.pth'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        if torch.cuda.is_available():
            self.model.cuda()
        print(self.model, '\n\n\n')


    def loss_fn(self, recon_x, x, mu, logvar):

        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        BCE = F.mse_loss(recon_x, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD


    def train(self, epoch):



        optimizer = optim.Adam(self.model.parameters(), lr=0.001)



        self.model.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_data_loader):
            data = data.cuda()
            optimizer.zero_grad()

            recon_batch, latent, mu, log_var = self.model(data)
            loss, bce, kld = self.loss_fn(recon_batch, data, mu, log_var)


            beta = epoch / 75 if epoch < 15 else 1
            loss = bce + kld

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 12 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE: {:.5f}, \tKLD: {:.5f},  \tSSIM: {:.5f}, \ttotal_Loss: {:.5f}'.format(
                    epoch, batch_idx * len(data), len(self.train_data_loader.dataset),
                           batch_idx * len(data)/len(self.train_data_loader.dataset)*100, bce, kld, 0, loss.item() / len(data)))


        with torch.no_grad():
            z = torch.randn(64, self.z_dim).cuda()
            sample = self.model.decode(z)

            save_image(sample.view(64, self.in_channel, self.input_size[0], self.input_size[1]), './result/sample_{}'.format(epoch) + '.png', normalize=True)
            #save_image(recon_batch[:64,].view(64, self.in_channel, self.input_size[0], self.input_size[1]), './result/train_{}'.format(epoch) + '.png')


        torch.save(self.model.state_dict(), self.save_model_name)
        print("mean, std : ", latent[0].mean().data, latent[0].std().data)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_data_loader.dataset)))




    def test(self, epoch):

        self.model.eval()

        with torch.no_grad():
            for data, _ in self.test_data_loader:
                data = data.cuda()
                recon, _, mu, log_var = self.model(data)

                save_image(recon.view(64, self.in_channel, self.input_size[0], self.input_size[1]), './result/test_{}'.format(epoch) + '.png', normalize=True)
                break

        self.eval_model_attention(eval_only=False, epoch=epoch)




    def eval_model_attention(self, eval_only, epoch=None):


        test_data_loader = load_data('mnist', 1, self.input_size,
                                     data_dir=self.test_dir,
                                     mode='test',
                                     color_channel=self.in_channel)



        if eval_only:
            self.model.load_state_dict(torch.load(self.save_model_name))
            self.model.to(self.device)

        # Make sure to call this !!!
        self.model.eval()


        #layer = get_encoder_layer(self.model, self.att_layer)
        layer = self.model.encoder[3]
        att   = VAE_attention(arch=self.model, target_layer=layer)

        if not eval_only:
            for i, data in enumerate(test_data_loader):


                print(i)
                images_a = data[0].to(self.device)
                images_r, attention_map = att(images_a[0].unsqueeze(0))


                plot_result_with_attention_one_image(images_a[0], images_r[0], attention_map.mean(0),'./result/train_test_{}-{}.png'.format(epoch, i))
                plot_with_attention_grid(attention_map, images_a[0], './result/test_channel_att_{}-{}_grid.png'.format(epoch, i))
                print()
                if i > 4:
                    break




        else:
            for i, data in enumerate(test_data_loader):

                print(i)
                images_a = data[0].to(self.device)
                images_r, attention_map = att(images_a[0].unsqueeze(0))

                plot_result_with_attention_one_image(images_a[0], images_r[0], attention_map.mean(0), './result/fig_test_{}.png'.format(i))
                plot_with_attention_grid(attention_map, images_a[0], './result/fig_test_{}_grid.png'.format(i))
                print()

                if i > 150:
                    break











mnist_setting = {'name': 'mnist',
                 'image_shape': [28, 28, 1],
                 'num_class': 10,

                 'batch_size': 128,
                 'h_dim': 1024,
                 'z_dim': 32,

                 'train_dir': './dataset/MNIST - anomaly2/training',
                 'test_dir': './dataset/MNIST - anomaly2/testing'
                 }



if __name__ == '__main__':

    vae = Model(mnist_setting)
    vae.eval_model_attention(eval_only=True)
    for epoch in range(1, 301):
       vae.train(epoch)
       #vae.test(epoch)
       print()

    vae.eval_model_attention(eval_only=True)






