import torch
import torch.nn as nn
import torch.nn.functional as F


# part 1.
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


# Part 2.
class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.Gelu = nn.GELU()
        if dropout_rate > 0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.Gelu(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out


# Part 3.
class MatrixGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(MatrixGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        feat = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return feat


# Part 4.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5
        self.query = MatrixGeneral((in_dim,), (self.heads, self.head_dim))  # 768 in_dims, 12 heads, 64 head dims.
        self.key = MatrixGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = MatrixGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = MatrixGeneral((self.heads, self.head_dim), (in_dim,))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return out


# Part 5.
class EncoderBasicBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout=0.1):
        super(EncoderBasicBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.multi_head_att = MultiHeadSelfAttention(in_dim, heads=num_heads, dropout_rate=attention_dropout)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.layer_norm2 = nn.LayerNorm(in_dim)
        self.mlp = MLPBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.multi_head_att(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.layer_norm2(out)
        out = self.mlp(out)
        out += residual
        return out


# Part 6.
class TransformerEncoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1,
                 attention_dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # position embedding.
        self.pos_embedding = PositionEmbedding(num_patches, emb_dim, dropout_rate)

        # encoder blocks.
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBasicBlock(in_dim, mlp_dim, num_heads, dropout_rate, attention_dropout)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        return out


# Part 7.
class VisionTransformerRebuild(nn.Module):
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attention_dropout=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformerRebuild, self).__init__()
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        # embedding layer
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fw, fw))
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = TransformerEncoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout)

        # classifier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # n, c, gh, gw
        emb = emb.permute(0, 2, 3, 1)  # n, gh, gw, c
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepare class token.
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feature = self.transformer(emb)

        # classifier
        logits = self.classifier(feature[:, 0])
        return logits


if __name__ == '__main__':
    model = VisionTransformerRebuild(num_layers=2)
    x = torch.randn((2, 3, 256, 256))
    out = model(x)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
