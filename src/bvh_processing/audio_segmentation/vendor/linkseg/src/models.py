import torch
import torchaudio
from torch import nn
from modules import *



class LinkSeg(nn.Module):
    def __init__(self, 
                encoder, 
                nb_ssm_classes=3, 
                nb_section_labels=7, 
                hidden_size=32, 
                output_channels=16,
                dropout_gnn=.1,
                dropout_cnn=.2,
                dropout_egat=.5,
                max_len=1500):
        super(LinkSeg, self).__init__()

        # frame encoder
        self.encoder = encoder
        # feature smoothing
        self.gnn = GCN_DENSE(in_size=hidden_size, hid_size=hidden_size, dropout=dropout_gnn)
        self.mlp = nn.Linear(hidden_size, hidden_size)
        # link feature extractor
        self.conv = ConvNetSSM(input_channels=1, output_channels=output_channels, shape=5, dropout=dropout_cnn)
        # positional embedding & linear projection
        self.pos_embedding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(max_len, output_channels)))
        self.fc = nn.Sequential(nn.Linear(output_channels, output_channels))
        # link classifier
        self.final_projection = nn.Linear(output_channels, nb_ssm_classes)
        # graph attention net
        self.gnn_final = EGAT(in_size=hidden_size, feat_size=output_channels, heads=8, feat_dropout=dropout_egat, attn_dropout=dropout_egat)
        # prediction heads
        self.bound_predictor = nn.Linear(hidden_size*2+output_channels, 1)
        self.class_predictor = nn.Linear(hidden_size, nb_section_labels)
        

    def forward(self, x):
        # frame encoding
        x_encoded = self.encoder(x)
        
        # self-similarity calculation (A)
        N = x_encoded.size(0)
        a = F.normalize(x_encoded, p=2, dim=-1)
        dist = torch.cdist(a, a, p=2).pow(2)
        std = torch.std(dist)
        gamma = -1/(2*std)
        A = dist.mul(gamma).exp()
        
        # feature smoothing (X')
        x_encoded = self.mlp(self.gnn(A, x_encoded))

        # self-similarity calculation (A')
        a = F.normalize(x_encoded, p=2, dim=-1)
        A = torch.cdist(a, a, p=2).pow(2)
        A_conv = self.conv(A.unsqueeze(0)).squeeze(0)
        A_conv = A_conv.permute(1, 2, 0)
        
        # positional embedding (E')
        A = torch.ones((N, N), device=x.device)
        src, dst = torch.nonzero(A, as_tuple=True)
        diff = torch.abs(src-dst) 
        x_time = self.pos_embedding[diff]
        x_pairwise = x_time.reshape(N, N, -1)
        A_conv = self.fc(A_conv) + x_pairwise

        # link prediction 
        A_pred = self.final_projection(A_conv).squeeze()

        # graph attention network
        edge_feat = A_conv[src, dst]
        g = dgl.graph((src, dst), num_nodes=N)
        x_gnn_final = self.gnn_final(g, x_encoded, edge_feat)

        # boundary prediction
        src_bounds = torch.arange(N-1)
        dst_bounds = torch.arange(1, N)
        link_bounds = A_conv[src_bounds, dst_bounds]
        x_bound = torch.cat((x_gnn_final[src_bounds], x_gnn_final[dst_bounds], link_bounds), -1)
        bound_pred = F.sigmoid(self.bound_predictor(x_bound)).squeeze()
        
        # class prediction
        class_pred = self.class_predictor(x_gnn_final)

        
        return x_gnn_final, bound_pred, class_pred, A_pred
    




class FrameEncoder(nn.Module):
    """
    Audio encoder, adapted from https://github.com/minzwon/semi-supervised-music-tagging-transformer.
    Copyright (c) 2021 ByteDance. Code developed by Minz Won.
    """
    def __init__(
        self,
        n_mels=64,
        conv_ndim=32,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_embedding=64,
        f_min=0,
        f_max=11025,
        dropout=0,
        hidden_dim = 32*2,
        attention_ndim = 32*2,
        attention_nlayers = 2,
        attention_nheads = 8,
    ):
        super(FrameEncoder, self).__init__()

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                     n_fft=n_fft,
                                                                     f_min=f_min,
                                                                     f_max=f_max,
                                                                     n_mels=n_mels, 
                                                                     hop_length=hop_length,
                                                                     power=2)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.frontend = ResFrontEnd(conv_ndim=conv_ndim, nharmonics=1, nmels=n_mels, output_size=attention_ndim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_embedding // 4, attention_ndim))

        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // 2, #,
            attention_ndim,
            dropout,
        )

        # projection
        self.mlp_head = nn.Linear(attention_ndim, hidden_dim)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (batch, time)

        Returns:
            x (torch.Tensor): (batch, hidden_dim)

        """
        # Input preprocessing
        x = self.spec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)

        # Input embedding
        x = self.frontend(x)
        x += self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        # transformer
        x = self.transformer(x)

        # pooling
        x = x.mean(1).squeeze(1)

        # projection
        x = self.mlp_head(x)

        return x