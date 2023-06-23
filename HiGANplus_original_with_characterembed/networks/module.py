import numpy as np
import torch
from torch import nn
import functools
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights
from .Transformer_model import *
import torchvision.models as models
from networks.sdt_transformer import *
class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn'):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])
        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.layer_name_mapping = {
            '9': "feat2",
            '13': "feat3",
            '16': "feat4",
        }

        self.cnn_ctc = nn.Sequential(
            nn.ReLU(),
            Conv2dBlock(df, df, 3, 1, 0,
                        norm=norm,
                        activation='relu')
        )
        if init != 'none':
            init_weights(self, init)

    def forward(self, x, ret_feats=False):
        feats = []
        for name, layer in self.cnn_backbone._modules.items():
            x = layer(x)
            if ret_feats and name in self.layer_name_mapping:
                feats.append(x)

        out = self.cnn_ctc(x).squeeze(-2)

        return out, feats


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02'):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim

        ######################################
        # Construct StyleEncoder
        ######################################
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)
        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_style(style)
        mu = self.mu(style)

        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class StyleEncoder_character_style(nn.Module):

    def __init__(self, style_dim=256, in_dim=7680, in_dim_style_char= 768 , init='N02'): ### todo: style_dim = 32, 7680
        super(StyleEncoder_character_style, self).__init__()
        self.style_dim = style_dim

        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 256 #7680
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        #print('mu')
        #print(in_dim, style_dim)
        self.mu = nn.Linear(in_dim_style_char, style_dim)
        self.logvar = nn.Linear(in_dim_style_char, style_dim)
        if init != 'none':
            init_weights(self, init)

        #### todo: character feature extraction
        # Define your model and hyperparameters
        # Prepare your data
        input_dim = 100
        output_dim = 256
        number_elements = 1000
        num_char = 15
        INP_CHANNEL = 1
        TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, TN_ENC_LAYERS = 512, 8, 512, 0.1, 1
        self.Feat_Encoder_char = nn.Sequential(*([nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] +
                                                 list(models.resnet18(pretrained=True).children())[1:-1]))
        self.encoder_layer_char = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,TN_DROPOUT, "relu", True)
        self.encoder_norm_char = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder_transformer_char = TransformerEncoder(self.encoder_layer_char, TN_ENC_LAYERS, self.encoder_norm_char)
        self.embedding_layer_char = nn.Linear(512, output_dim)
        #self.model_transformer_char = FeatureTransformer(input_dim=input_dim, output_dim=output_dim, num_layers=2, num_heads=2, hidden_dim=64)
        #### todo: character feature extraction
        self.norm2 = nn.LayerNorm(256)

        self.mta_model = torch.nn.MultiheadAttention(512, 8)
    def forward(self, img, img_len, imgs_char, num_char_in_words, cnn_backbone=None, ret_feats=False, vae_mode=False):
        ### todo: character embedding
        #print('char2')
        #print(num_char_in_words)
        B, N, R, C = imgs_char.shape
        FEAT_ST = self.Feat_Encoder_char(imgs_char.view(B * N, 1, R, C))
        FEAT_ST_ENC = FEAT_ST.view(B, N, 512)  # (B, 512, 1, -1), (B, N*512) ### todo: put number of images same as
        ### todo: add src_mask here
        #src_mask = src_mask.squeeze(dim =1)
        #src_mask = einops.repeat(src_mask, 'b h w -> (repeat b) h w', repeat=8)
        #attn_mask_3d = torch.repeat_interleave(src_mask, 8, dim=0)  # [N * H, T, S]
        memory = self.encoder_transformer_char(FEAT_ST_ENC, attn_mask_3d)
        #### todo: zero-out the unused keys here
        #for batch_index in range(int(memory.shape[0])):
            #print(batch_index, int(num_char_in_words[batch_index]))
        #    memory[batch_index,int(num_char_in_words[batch_index]):,:] = 0
        #query = key = value = FEAT_ST_ENC
        #memory = self.mta_model(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=attn_mask_3d)[0]
        #memory = memory.transpose(0, 1)
        #memory = memory.view(memory.shape[1], memory.shape[0], memory.shape[2])
        output_embedding_char = memory
        output_embedding_char = output_embedding_char.view(output_embedding_char.shape[0], output_embedding_char.shape[2], output_embedding_char.shape[1])
        #self.embedding_layer_char(memory)
        #output_embedding_char = output_embedding_char.squeeze()
        ######
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        #print(img_len)
        #print(feat.size(-1))
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        #print(feat.shape, img_len_mask.shape)
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        #print(style.shape)
        #style = self.linear_style(style)
        #style = self.norm2(style)
        ### todo: character embedding concatenate with word features
        #output_embedding_char = output_embedding_char.reshape(output_embedding_char.shape[0], output_embedding_char.shape[1]*output_embedding_char.shape[2])
        #num_char_in_words = num_char_in_words.unsqueeze(1)
        #print(num_char_in_words, output_embedding_char.size(-1))
        character_len_mask = _len2mask(num_char_in_words, output_embedding_char.size(-1)).unsqueeze(1).float().detach()
        style_output_embedding_char = (output_embedding_char * character_len_mask).sum(dim=-1) / (num_char_in_words.unsqueeze(1).float() + 1e-8)
        #print(num_char_in_words.shape,output_embedding_char.size(-1), character_len_mask)
        #print('masked output embedding')
        #print(output_embedding_char.shape)
        #print('style:', style.shape)
        #print('char style:', style_output_embedding_char.shape)
        overall_style_features = torch.cat((style, style_output_embedding_char), dim = 1)#style = output_embedding_char #overall_style_features

        style = overall_style_features
        ### todo: character embedding concatenate with word features
        #style = output_embedding_char
        #style = self.linear_style(style)
        #print('overall:', style.shape)
        mu = self.mu(style)
        #print('embed shape:', mu.shape)
        #print('done mu')
        vae_mode = True
        if vae_mode:
            logvar = self.logvar(style)
            #print(logvar.shape)
            style = self.reparameterize(mu, logvar)
            #print(style.shape)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            #print("**************** returning feats ************************")
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class StyleEncoder_char_sep_attention_character_style(nn.Module):

    def __init__(self, style_dim=32, in_dim=7680, in_dim_style_char=512,
                 init='N02'):  ### todo: style_dim = 32, 7680
        super(StyleEncoder_char_sep_attention_character_style, self).__init__()
        self.style_dim = style_dim
        in_dim_style_char = 512

        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 512  # 7680
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        # print('mu')
        # print(in_dim, style_dim)
        self.mu_word = nn.Linear(in_dim_style_char, style_dim)
        self.mu_char = nn.Linear(in_dim_style_char, 16)
        self.mu_overall = nn.Linear(48, 32)
        self.linear_size_match = nn.Linear(256, 512)
        self.char_norm = nn.LayerNorm(512)
        self.word_norm = nn.LayerNorm(512)
        self.logvar = nn.Linear(48, style_dim)
        if init != 'none':
            init_weights(self, init)

        #### todo: character feature extraction
        # Define your model and hyperparameters
        # Prepare your data
        input_dim = 100
        output_dim = 256
        number_elements = 1000
        num_char = 15
        INP_CHANNEL = 1
        TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, TN_ENC_LAYERS = 512, 8, 512, 0.1, 1
        self.Feat_Encoder_char = nn.Sequential(*(
                    [nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
                models.resnet18(pretrained=True).children())[1:-1]))
        self.encoder_layer_char = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT,
                                                          "relu", True)
        self.encoder_norm_char = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder_transformer_char = TransformerEncoder(self.encoder_layer_char, TN_ENC_LAYERS,
                                                           self.encoder_norm_char)
        self.embedding_layer_char = nn.Linear(512, output_dim)
        # self.model_transformer_char = FeatureTransformer(input_dim=input_dim, output_dim=output_dim, num_layers=2, num_heads=2, hidden_dim=64)
        #### todo: character feature extraction
        # self.norm2 = nn.LayerNorm(256)
        self.mta_model = torch.nn.MultiheadAttention(512, 8)

    def forward(self, img, img_len, imgs_char, num_char_in_words,cnn_backbone=None, ret_feats=False,
                vae_mode=False):
        ### todo: character embedding
        # print('char2')
        # print(num_char_in_words)
        B, N, R, C = imgs_char.shape
        FEAT_ST = self.Feat_Encoder_char(imgs_char.view(B * N, 1, R, C))
        FEAT_ST_ENC = FEAT_ST.view(B, N, 512)  # , 512)  # (B, 512, 1, -1), (B, N*512) ### todo: put number of images same as
        ### todo: add src_mask here
        # src_mask = src_mask.squeeze(dim =1)
        # src_mask = einops.repeat(src_mask, 'b h w -> (repeat b) h w', repeat=8)
        #attn_mask_3d = torch.repeat_interleave(src_mask, 8, dim=0)  # [N * H, T, S]
        ### todo: stack style features with memory features
        feat, all_feats = cnn_backbone(img, ret_feats)

        img_len = img_len // cnn_backbone.reduce_len_scale
        # print(img_len)
        # print('feat', img_len , feat.shape)
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        # print(feat.shape, img_len_mask.shape)
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        style = self.linear_size_match(style)
        word_style = style.unsqueeze(1)

        # style_memory_stack = torch.cat((style, FEAT_ST_ENC), dim=1)
        # print('style' , style.shape)
        # print(style.shape)
        # style = self.linear_style(style)
        # style = self.norm2(style)
        # , attn_mask_3d)
        # output_embedding_char = memory
        # print(memory.shape)
        # = torch.stack((style.unsqueeze(1), memory))
        # print('memory', memory.shape)
        # style_memory_stack = torch.cat((style, memory), dim=1)
        # print('style_memory_stack' , style_memory_stack.shape)
        #### todo: zero-out the unused keys here
        """
        for batch_index in range(int(FEAT_ST_ENC.shape[0])):
            # print(batch_index, int(num_char_in_words[batch_index]))
            if num_char_in_words[batch_index] > 14:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]) - 1, :] = style[batch_index, :, :]
            else:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]), :] = style[batch_index, :, :]
        """
        #print('Feat_enc', FEAT_ST_ENC.shape)
        output_embedding_char = self.encoder_transformer_char(FEAT_ST_ENC)
        # query = key = value = FEAT_ST_ENC
        # memory = self.mta_model(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=attn_mask_3d)[0]
        # memory = memory.transpose(0, 1)
        # memory = memory.view(memory.shape[1], memory.shape[0], memory.shape[2])
        # output_embedding_char = memory
        # print(num_char_in_words, output_embedding_char.shape)
        output_embedding_char = output_embedding_char.view(output_embedding_char.shape[0],
                                                           output_embedding_char.shape[2],
                                                           output_embedding_char.shape[1])  # self.embedding_layer_char(memory)
        # output_embedding_char = output_embedding_char.squeeze()
        ### todo: character embedding concatenate with word features
        # output_embedding_char = output_embedding_char.reshape(output_embedding_char.shape[0], output_embedding_char.shape[1]*output_embedding_char.shape[2])
        # num_char_in_words = num_char_in_words.unsqueeze(1)
        # print(num_char_in_words)
        #num_char_in_words = num_char_in_words + 1
        # print(num_char_in_words, output_embedding_char.shape)
        # print(num_char_in_words)
        character_len_mask = _len2mask(num_char_in_words, output_embedding_char.size(-1)).unsqueeze(1).float().detach()
        char_style = (output_embedding_char * character_len_mask).sum(dim=-1) / (num_char_in_words.unsqueeze(1).float() + 1e-8)
        word_style = word_style.squeeze(1)
        #norm_word = self.word_norm(word_style)
        #norm_char = self.char_norm(char_style)
        #style = torch.cat((word_style, char_style), dim=1)
        # print(num_char_in_words.shape,output_embedding_char.size(-1), character_len_mask)
        # print('masked output embedding')
        # print(output_embedding_char.shape)
        # style = style.unsqueeze(1)
        # print('style', style.shape)
        # print(style_output_embedding_char.shape)
        # overall_style_features = torch.cat((style, style_output_embedding_char), dim = 1)#style = output_embedding_char #overall_style_features
        # style = overall_style_features
        ### todo: character embedding concatenate with word features
        # style = output_embedding_char
        # style = self.linear_style(style)
        # print(style.shape)
        #print('mu_word:', word_style.size())
        #print('mu_char:', char_style.size())
        mu_word = self.mu_word(word_style)
        mu_char = self.mu_char(char_style)
        #print('mu_word:', mu_word.size())
        #print('mu_char:', mu_char.size())
        style = torch.cat((mu_word, mu_char), dim = 1)
        mu = self.mu_overall(style)
        #print('concat:', concat.size())

        #print('mu:', concat.size())
        # print('mu', mu.shape)
        # print('done mu')
        #vae_mode = True
        if vae_mode:
            logvar = self.logvar(style)
            # print(logvar.shape)
            style = self.reparameterize(mu, logvar)
            # print(style.shape)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            # print("**************** returning feats ************************")
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar) ### todo: uncomment
        eps = torch.randn_like(std)
        return eps * std + mu


class StyleEncoder_w_c_character_style(nn.Module):

    def __init__(self, style_dim=32, in_dim=7680, in_dim_style_char=512,
                 init='N02', d_model=256, nhead=8, num_encoder_layers=2, num_head_layers=1,
                 wri_dec_layers=2, gly_dec_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, return_intermediate_dec=True):  ### todo: style_dim = 32, 7680
        super(StyleEncoder_w_c_character_style, self).__init__()
        self.style_dim = style_dim
        in_dim_style_char = 512

        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 512  # 7680
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        # print('mu')
        # print(in_dim, style_dim)
        self.mu = nn.Linear(in_dim_style_char, style_dim)
        self.linear_size_match = nn.Linear(256, 512)
        self.logvar = nn.Linear(in_dim_style_char, style_dim)
        if init != 'none':
            init_weights(self, init)

        #### todo: character feature extraction
        # Define your model and hyperparameters
        # Prepare your data
        input_dim = 100
        output_dim = 256
        number_elements = 1000
        num_char = 15
        INP_CHANNEL = 1
        TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, TN_ENC_LAYERS = 512, 8, 512, 0.1, 1
        """
        self.Feat_Encoder_char = nn.Sequential(*(
                    [nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
                models.resnet18(pretrained=True).children())[1:-1]))
        self.encoder_layer_char = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT,
                                                          "relu", True)
        self.encoder_norm_char = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder_transformer_char = TransformerEncoder(self.encoder_layer_char, TN_ENC_LAYERS,
                                                           self.encoder_norm_char)
        self.embedding_layer_char = nn.Linear(512, output_dim)
        # self.model_transformer_char = FeatureTransformer(input_dim=input_dim, output_dim=output_dim, num_layers=2, num_heads=2, hidden_dim=64)
        """
        ### todo: code ftom sdt_transformer paper
        self.Feat_Encoder_char = nn.Sequential(*(
                [nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
            models.resnet18(pretrained=True).children())[1:-1]))
        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        ####
        #### todo: character feature extraction
        # self.norm2 = nn.LayerNorm(256)
        self.mta_model = torch.nn.MultiheadAttention(512, 8)

    def forward(self, img, img_len, imgs_char, num_char_in_words,cnn_backbone=None, ret_feats=False,
                vae_mode=False):
        ### todo: character embedding
        # print('char2')
        # print(num_char_in_words)
        B, N, R, C = imgs_char.shape
        FEAT_ST = self.Feat_Encoder_char(imgs_char.view(B * N, 1, R, C))
        FEAT_ST_ENC = FEAT_ST.view(B, N, 512)  # , 512)  # (B, 512, 1, -1), (B, N*512) ### todo: put number of images same as

        ### todo: add src_mask here
        # src_mask = src_mask.squeeze(dim =1)
        # src_mask = einops.repeat(src_mask, 'b h w -> (repeat b) h w', repeat=8)
        #attn_mask_3d = torch.repeat_interleave(src_mask, 8, dim=0)  # [N * H, T, S]
        ### todo: stack style features with memory features
        feat, all_feats = cnn_backbone(img, ret_feats)

        img_len = img_len // cnn_backbone.reduce_len_scale
        # print(img_len)
        # print('feat', img_len , feat.shape)
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        # print(feat.shape, img_len_mask.shape)
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)

        style = self.linear_size_match(style)
        style = style.unsqueeze(1)
        # style_memory_stack = torch.cat((style, FEAT_ST_ENC), dim=1)
        # print('style' , style.shape)
        # print(style.shape)
        # style = self.linear_style(style)
        # style = self.norm2(style)
        # , attn_mask_3d)
        # output_embedding_char = memory
        # print(memory.shape)
        # = torch.stack((style.unsqueeze(1), memory))
        # print('memory', memory.shape)
        # style_memory_stack = torch.cat((style, memory), dim=1)
        # print('style_memory_stack' , style_memory_stack.shape)
        #### todo: zero-out the unused keys here
        for batch_index in range(int(FEAT_ST_ENC.shape[0])):
            # print(batch_index, int(num_char_in_words[batch_index]))
            if num_char_in_words[batch_index] > 14:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]) - 1, :] = style[batch_index, :, :]
            else:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]), :] = style[batch_index, :, :]
        #print('Feat_enc', FEAT_ST_ENC.shape)
        ### todo: add positional embedding with features from characters and images
        output_embedding_char = self.encoder_transformer_char(FEAT_ST_ENC)
        # query = key = value = FEAT_ST_ENC
        # memory = self.mta_model(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=attn_mask_3d)[0]
        # memory = memory.transpose(0, 1)
        # memory = memory.view(memory.shape[1], memory.shape[0], memory.shape[2])
        # output_embedding_char = memory
        # print(num_char_in_words, output_embedding_char.shape)
        output_embedding_char = output_embedding_char.view(output_embedding_char.shape[0],
                                                           output_embedding_char.shape[2],
                                                           output_embedding_char.shape[1])  # self.embedding_layer_char(memory)
        # output_embedding_char = output_embedding_char.squeeze()
        ######

        ### todo: character embedding concatenate with word features
        # output_embedding_char = output_embedding_char.reshape(output_embedding_char.shape[0], output_embedding_char.shape[1]*output_embedding_char.shape[2])
        # num_char_in_words = num_char_in_words.unsqueeze(1)
        # print(num_char_in_words)

        num_char_in_words = num_char_in_words + 1
        # print(num_char_in_words, output_embedding_char.shape)
        # print(num_char_in_words)
        character_len_mask = _len2mask(num_char_in_words, output_embedding_char.size(-1)).unsqueeze(1).float().detach()
        style = (output_embedding_char * character_len_mask).sum(dim=-1) / (num_char_in_words.unsqueeze(1).float() + 1e-8)
        # print(num_char_in_words.shape,output_embedding_char.size(-1), character_len_mask)
        # print('masked output embedding')
        # print(output_embedding_char.shape)
        # style = style.unsqueeze(1)
        # print('style', style.shape)
        # print(style_output_embedding_char.shape)
        # overall_style_features = torch.cat((style, style_output_embedding_char), dim = 1)#style = output_embedding_char #overall_style_features
        # style = overall_style_features
        ### todo: character embedding concatenate with word features
        # style = output_embedding_char
        # style = self.linear_style(style)
        # print(style.shape)
        mu = self.mu(style)
        # print('mu', mu.shape)
        # print('done mu')
        vae_mode = True
        if vae_mode:
            logvar = self.logvar(style)
            # print(logvar.shape)
            style = self.reparameterize(mu, logvar)
            # print(style.shape)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            # print("**************** returning feats ************************")
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class StyleEncoder_transformer(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02', d_model=256, nhead=8, num_encoder_layers=2,
                 num_head_layers=1, wri_dec_layers=2,
                 gly_dec_layers=2, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True,
                 return_intermediate_dec=True):
        super(StyleEncoder_transformer, self).__init__()
        self.style_dim = style_dim
        self.Feat_Encoder = nn.Sequential(
            *([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
                models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 256
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode = False):

        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        ### todo: add transformer layer here print
        # print('style:', style.size())
        style = style.unsqueeze(1)
        FEAT_ST_ENC = self.add_position(style)
        # print('positional encoding:', FEAT_ST_ENC.size())
        memory = self.base_encoder(FEAT_ST_ENC)
        # print('memory:', memory.size())
        B, N, C = memory.shape
        style = memory.view(B, N * C)
        # print('reshape memory style', style.size())

        style = self.linear_style(style)

        # print('memory style', style)
        mu = self.mu(style)
        # print('mu', mu.size())
        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            # print('final style', style.size())
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        # print('mu', mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print('eps', eps)
        return eps * std + mu

class StyleEncoder_char_word_transformer(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02', d_model=256, nhead=8, num_encoder_layers=2,
                 num_head_layers=1, wri_dec_layers=2,
                 gly_dec_layers=2, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True,
                 return_intermediate_dec=True):
        super(StyleEncoder_char_word_transformer, self).__init__()
        self.style_dim = style_dim
        self.Feat_Encoder = nn.Sequential(
            *([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 256
        char_in_dim = 32
        word_in_dim = 32
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        self.char_linear_style = nn.Sequential(
            nn.Linear(in_dim, char_in_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=char_in_dim),
        )
        self.word_linear_style = nn.Sequential(
            nn.Linear(in_dim, word_in_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=word_in_dim),
        )
        #self.char_in_dim = nn.Linear(in_dim, char_in_dim)
        #self.word_in_dim = nn.Linear(in_dim, word_in_dim)
        self.mu = nn.Linear(style_dim, style_dim)
        self.logvar = nn.Linear(style_dim, style_dim)

        self._reset_parameters()
        ######################################
        # Construct word StyleEncoder
        ######################################
        # todo: normal weights initialization for word style encoder layer
        #if init != 'none':
        #    init_weights(self, init)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):

        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        ### todo: add transformer layer here print
        # print('style:', style.size())
        style = style.unsqueeze(1)
        FEAT_ST_ENC = self.add_position(style)
        #print('positional encoding:', FEAT_ST_ENC.size())
        memory = self.base_encoder(FEAT_ST_ENC)
        #print('memory:', memory.size())
        B, N, C = memory.shape
        char_style = memory.view(B, N * C)
        char_style = self.linear_style(char_style)
        #print('char shape', char_style.size())


        #### todo: word style embedding
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        word_style = self.linear_style(style)
        #print('word shape', word_style.size())
        char_embed = self.char_linear_style(char_style)
        word_embed = self.word_linear_style(word_style)
        #print('char shape', char_embed.size())
        #print('word shape', word_embed.size())
        #style = torch.cat((word_embed, char_embed), dim=1)
        ## todo: sum

        style = (word_embed + char_embed)
        #print('style', style.size())
        mu = self.mu(style)
        # print('mu', mu.size())
        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            # print('final style', style.size())
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        # print('mu', mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print('eps', eps)
        return eps * std + mu

class StyleEncoder_char_word_transformer_composite(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02', d_model=256, nhead=8, num_encoder_layers=2,
                 num_head_layers=1, wri_dec_layers=2,
                 gly_dec_layers=2, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True,
                 return_intermediate_dec=True):
        super(StyleEncoder_char_word_transformer, self).__init__()
        self.style_dim = style_dim
        self.Feat_Encoder = nn.Sequential(
            *([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(
                models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 256
        char_in_dim = 32
        word_in_dim = 32
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        self.char_in_dim = nn.Linear(in_dim, char_in_dim)
        self.word_in_dim = nn.Linear(in_dim, word_in_dim)
        self.mu = nn.Linear(48, style_dim)
        self.logvar = nn.Linear(48, style_dim)

        self._reset_parameters()
        ######################################
        # Construct word StyleEncoder
        ######################################
        # todo: normal weights initialization for word style encoder layer
        # if init != 'none':
        #    init_weights(self, init)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):

        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        ### todo: add transformer layer here print
        # print('style:', style.size())
        style = style.unsqueeze(1)
        FEAT_ST_ENC = self.add_position(style)
        # print('positional encoding:', FEAT_ST_ENC.size())
        memory = self.base_encoder(FEAT_ST_ENC)
        # print('memory:', memory.size())
        B, N, C = memory.shape
        char_style = memory.view(B, N * C)
        char_style = self.linear_style(char_style)
        # print('char shape', char_style.size())

        #### todo: word style embedding
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        word_style = self.linear_style(style)
        # print('word shape', word_style.size())
        char_embed = self.char_in_dim(char_style)
        word_embed = self.word_in_dim(word_style)
        # print('char shape', char_embed.size())
        # print('word shape', word_embed.size())
        ### todo: add a copmposite function here instead of concat


        ### todo: finish a copmposite function here instead of concat
        #style = torch.cat((word_embed, char_embed), dim=1)
        # print('style', style.size())
        mu = self.mu(style)
        # print('mu', mu.size())
        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            # print('final style', style.size())
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        # print('mu', mu)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print('eps', eps)
        return eps * std + mu
class StyleEncoder_char_word_attention_style(nn.Module):

    def __init__(self, style_dim=32, in_dim=7680, in_dim_style_char=512,
                 init='xavier', d_model=512, nhead=8, num_encoder_layers=2, num_head_layers=1,
                 wri_dec_layers=2, gly_dec_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, return_intermediate_dec=True):  ### todo: style_dim = 32, 7680 init = N02
        super(StyleEncoder_char_word_attention_style, self).__init__()
        self.style_dim = style_dim
        in_dim_style_char = 512

        ######################################
        # Construct StyleEncoder
        ######################################
        in_dim = 512  # 7680
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )
        # print('mu')
        # print(in_dim, style_dim)
        self.mu = nn.Linear(in_dim_style_char, style_dim)
        self.linear_size_match = nn.Linear(256, 512)
        self.logvar = nn.Linear(in_dim_style_char, style_dim)
        #### todo: character feature extraction
        # Define your model and hyperparameters
        # Prepare your data
        input_dim = 100
        output_dim = 256
        number_elements = 1000
        num_char = 15
        INP_CHANNEL = 1
        TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT, TN_ENC_LAYERS = 512, 8, 512, 0.1, 1

        self.Feat_Encoder_char = nn.Sequential(*(
                    [nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-1]))
        """
        self.encoder_layer_char = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD, TN_DROPOUT,
                                                          "relu", True)
        self.encoder_norm_char = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder_transformer_char = TransformerEncoder(self.encoder_layer_char, TN_ENC_LAYERS,
                                                           self.encoder_norm_char)
        self.embedding_layer_char = nn.Linear(512, output_dim)
        # self.model_transformer_char = FeatureTransformer(input_dim=input_dim, output_dim=output_dim, num_layers=2, num_heads=2, hidden_dim=64)
        """
        ### todo: code from sdt_transformer paper
        #self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)
        #### todo: character feature extraction
        # self.norm2 = nn.LayerNorm(256)
        #self.mta_model = torch.nn.MultiheadAttention(512, 8)
        #self._reset_parameters()
        if init != 'none':
            init_weights(self, init)
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img, img_len, imgs_char, num_char_in_words,cnn_backbone=None, ret_feats=False, vae_mode=False):
        ### todo: character embedding
        # print('char2')
        # print(num_char_in_words)
        B, N, R, C = imgs_char.shape
        FEAT_ST = self.Feat_Encoder_char(imgs_char.view(B * N, 1, R, C))
        FEAT_ST_ENC = FEAT_ST.view(B, N, 512)  # , 512)  # (B, 512, 1, -1), (B, N*512) ### todo: put number of images same as
        #print('Feat_ST_ENC', FEAT_ST_ENC.shape)
        ### todo: add src_mask here
        #src_mask = src_mask.squeeze(dim =1)
        #src_mask = einops.repeat(src_mask, 'b h w -> (repeat b) h w', repeat=8)
        #attn_mask_3d = torch.repeat_interleave(src_mask, 8, dim=0)  # [N * H, T, S]
        ### todo: stack style features with memory features
        feat, all_feats = cnn_backbone(img, ret_feats)

        img_len = img_len // cnn_backbone.reduce_len_scale
        # print(img_len)
        # print('feat', img_len , feat.shape)
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        # print(feat.shape, img_len_mask.shape)
        style = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)

        style = self.linear_size_match(style)
        style = style.unsqueeze(1)
        # style_memory_stack = torch.cat((style, FEAT_ST_ENC), dim=1)
        # print('style' , style.shape)
        # print(style.shape)
        # style = self.linear_style(style)
        # style = self.norm2(style)
        # , attn_mask_3d)
        # output_embedding_char = memory
        # print(memory.shape)
        # = torch.stack((style.unsqueeze(1), memory))
        # print('memory', memory.shape)
        # style_memory_stack = torch.cat((style, memory), dim=1)
        # print('style_memory_stack' , style_memory_stack.shape)
        #### todo: zero-out the unused keys here
        for batch_index in range(int(FEAT_ST_ENC.shape[0])):
            # print(batch_index, int(num_char_in_words[batch_index]))
            if num_char_in_words[batch_index] > 14:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]) - 1, :] = style[batch_index, :, :]
            else:
                FEAT_ST_ENC[batch_index, int(num_char_in_words[batch_index]), :] = style[batch_index, :, :]
        #print('Feat_enc', FEAT_ST_ENC.shape)
        ### todo: add positional embedding with features from characters and images
        FEAT_ST_ENC = self.add_position(FEAT_ST_ENC)
        #print('positional encoding:', FEAT_ST_ENC.size())
        output_embedding_char = self.base_encoder(FEAT_ST_ENC)
        #output_embedding_char = self.encoder_transformer_char(FEAT_ST_ENC)
        # query = key = value = FEAT_ST_ENC
        # memory = self.mta_model(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), attn_mask=attn_mask_3d)[0]
        # memory = memory.transpose(0, 1)
        # memory = memory.view(memory.shape[1], memory.shape[0], memory.shape[2])
        # output_embedding_char = memory
        # print(num_char_in_words, output_embedding_char.shape)
        output_embedding_char = output_embedding_char.view(output_embedding_char.shape[0],
                                                           output_embedding_char.shape[2],
                                                           output_embedding_char.shape[1])  # self.embedding_layer_char(memory)
        # output_embedding_char = output_embedding_char.squeeze()
        ######

        ### todo: character embedding concatenate with word features
        # output_embedding_char = output_embedding_char.reshape(output_embedding_char.shape[0], output_embedding_char.shape[1]*output_embedding_char.shape[2])
        # num_char_in_words = num_char_in_words.unsqueeze(1)
        # print(num_char_in_words)

        num_char_in_words = num_char_in_words + 1
        #print(num_char_in_words, output_embedding_char.shape)
        # print(num_char_in_words)
        character_len_mask = _len2mask(num_char_in_words, output_embedding_char.size(-1)).unsqueeze(1).float().detach()
        style = (output_embedding_char * character_len_mask).sum(dim=-1) / (num_char_in_words.unsqueeze(1).float() + 1e-8)
        # print(num_char_in_words.shape,output_embedding_char.size(-1), character_len_mask)
        # print('masked output embedding')
        # print(output_embedding_char.shape)
        # style = style.unsqueeze(1)
        #print('style', style.shape)
        # print(style_output_embedding_char.shape)
        # overall_style_features = torch.cat((style, style_output_embedding_char), dim = 1)#style = output_embedding_char #overall_style_features
        # style = overall_style_features
        ### todo: character embedding concatenate with word features
        # style = output_embedding_char
        # style = self.linear_style(style)
        # print(style.shape)
        mu = self.mu(style)
        # print('mu', mu.shape)
        # print('done mu')
        if vae_mode:
            logvar = self.logvar(style)
            # print(logvar.shape)
            style = self.reparameterize(mu, logvar)
            # print(style.shape)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            # print("**************** returning feats ************************")
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=372, in_dim=256, init='N02'):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32

        ######################################
        # Construct WriterIdentifier
        ######################################

        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        wid_logits = self.linear_wid(wid_feat)
        if ret_feats:
            return wid_logits, all_feats
        else:
            return wid_logits

    def return_feat(self, img, img_len):
        feat = self.cnn_backbone(img)
        img_len = img_len // self.reduce_len_scale
        out_w = self.cnn_wid(feat).squeeze(-2)
        img_len_mask = _len2mask(img_len, out_w.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (out_w * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)
        for j in range(2):
            wid_feat = self.linear_wid[j](wid_feat)
        return wid_feat


class Recognizer(nn.Module):
    # resolution: 32  max_dim: 512  in_channel: 1  norm: 'none'  init: 'N02'  dropout: 0.  n_class: 72  rnn_depth: 0
    def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none',
                 init='none', rnn_depth=1, dropout=0.0, bidirectional=True):
        super(Recognizer, self).__init__()
        self.len_scale = 16
        self.use_rnn = rnn_depth > 0
        self.bidirectional = bidirectional

        ######################################
        # Construct Backbone
        ######################################
        nf = resolution
        cnn_f = [nn.ConstantPad2d(2, -1),
                 Conv2dBlock(in_channel, nf, 5, 2, 0,
                             norm='none',
                             activation='none')]
        for i in range(2):
            nf_out = min([int(nf * 2), max_dim])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2)]
            cnn_f += [nn.ZeroPad2d(1)]
            cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            nf = min([nf_out, max_dim])

        df = nf
        for i in range(2):
            df_out = min([int(df * 2), max_dim])
            cnn_f += [ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout)]
            cnn_f += [ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)]
            if i < 1:
                cnn_f += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = min([df_out, max_dim])

        ######################################
        # Construct Classifier
        ######################################
        cnn_c = [nn.ReLU(),
                 Conv2dBlock(df, df, 3, 1, 0,
                             norm=norm,
                             activation='relu')]

        self.cnn_backbone = nn.Sequential(*cnn_f)
        self.cnn_ctc = nn.Sequential(*cnn_c)
        if self.use_rnn:
            if bidirectional:
                self.rnn_ctc = DeepBLSTM(df, df, rnn_depth, bidirectional=True)
            else:
                self.rnn_ctc = DeepLSTM(df, df, rnn_depth)
        self.ctc_cls = nn.Linear(df, n_class)

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, x_len=None):
        cnn_feat = self.cnn_backbone(x)
        cnn_feat2 = self.cnn_ctc(cnn_feat)
        ctc_feat = cnn_feat2.squeeze(-2).transpose(1, 2)
        if self.use_rnn:
            if self.bidirectional:
                ctc_len = x_len // (self.len_scale + 1e-8)
            else:
                ctc_len = None
            ctc_feat = self.rnn_ctc(ctc_feat, ctc_len.cpu())
        logits = self.ctc_cls(ctc_feat)
        if self.training:
            logits = logits.transpose(0, 1).log_softmax(2)
            logits.requires_grad_(True)
        return logits

    def frozen_bn(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fix_bn)

