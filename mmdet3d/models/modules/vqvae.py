import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x



class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)

        )

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, out_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_dim, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices



class VQVAE(nn.Module):
    def __init__(self, feature_channel, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(feature_channel, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, feature_channel, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

        # # from IPython import embed
        # # embed(header='xxx')
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # filename = '/home/wangz/wangzhe21/VIMI/work_dirs/0515_VIMI_VQVAE_960x540_12e_bs2x1/best_car_3d_0.5_epoch_10.pth'
        # checkpoint = torch.load(filename, map_location = device)
        # state_dict = checkpoint['state_dict']

        # # 第一步：读取当前模型参数
        # encoder_model_dict = self.encoder.state_dict()
        # # 第二步：读取预训练模型
        # encoder_model_dict_new = {k.replace('vqvae.encoder.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.encoder.')}
        # # 第三步：使用预训练的模型更新当前模型参数
        # encoder_model_dict.update(encoder_model_dict_new)
        # # 第四步：加载模型参数
        # self.encoder.load_state_dict(encoder_model_dict)


        # pre_quantization_conv_model_dict = self.pre_quantization_conv.state_dict()
        # pre_quantization_conv_model_dict_new = {k.replace('vqvae.pre_quantization_conv.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.pre_quantization_conv.')}
        # pre_quantization_conv_model_dict.update(pre_quantization_conv_model_dict_new)
        # self.pre_quantization_conv.load_state_dict(pre_quantization_conv_model_dict)

        # vector_quantization_model_dict = self.vector_quantization.state_dict()
        # vector_quantization_model_dict_new = {k.replace('vqvae.vector_quantization.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.vector_quantization.')}
        # vector_quantization_model_dict.update(vector_quantization_model_dict_new)
        # self.vector_quantization.load_state_dict(vector_quantization_model_dict)


        # decoder_model_dict = self.decoder.state_dict()
        # decoder_model_dict_new = {k.replace('vqvae.decoder.', ''): v for k, v in state_dict.items() if k.startswith('vqvae.decoder.')}
        # decoder_model_dict.update(decoder_model_dict_new)
        # self.decoder.load_state_dict(decoder_model_dict)




    def forward(self, x, verbose=False):
        
        # from IPython import embed
        # embed(header= 'encoder')
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


class VQVAE_Veh(nn.Module):
    def __init__(self, feature_channel, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False, 
                 load_ckpt=False, ckpt_path=None, load_dict=None):
        super(VQVAE_Veh, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(feature_channel, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, feature_channel, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None



        # if load_ckpt and ckpt_path is not None:
        #     if not os.path.exists(ckpt_path):
        #         raise ValueError('ckpt_path does not exists')
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     filename = ckpt_path
        #     checkpoint = torch.load(filename, map_location = device)
        #     state_dict = checkpoint['state_dict']

        #     if load_dict != None:
        #         for i in range(len(load_dict)):
        #             # 第一步：读取当前模型参数
        #             encoder_model_dict = getattr(self,load_dict[i]).state_dict()
        #             module_str = 'vqvae_veh.'+ load_dict[i] + '.'
        #             # 第二步：读取预训练模型
        #             encoder_model_dict_new = {k.replace(module_str, ''): v for k, v in state_dict.items() if k.startswith(module_str)}
        #             # 第三步：使用预训练的模型更新当前模型参数
        #             encoder_model_dict.update(encoder_model_dict_new)
        #             # 第四步：加载模型参数
        #             getattr(self,load_dict[i]).load_state_dict(encoder_model_dict)


    def forward(self, x, verbose=False):
        
        # from IPython import embed
        # embed(header= 'encoder')
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity