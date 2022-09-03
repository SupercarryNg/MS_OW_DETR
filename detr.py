import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads
        assert (self.heads * self.head_dim == self.embed_size), "Embedding size need to be divided by head"

        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * self.heads, self.embed_size)

    def forward(self, queries, keys, values):
        N = queries.shape[0]
        queries_len, keys_len, values_len = queries.shape[1], keys.shape[1], values.shape[1]
        assert (queries.shape[2] == self.embed_size), "Input Size must equal to the embedding size"

        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        values = values.reshape(N, values_len, self.heads, self.head_dim)

        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        # query shape: (N, q_len, head, head_dim)
        # keys shape: (N, k_len, head, head_dim)
        # energy shape: (N, head, q_len, k_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)  # ***
        output = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, queries_len,
                                                                              self.heads * self.head_dim)
        output = self.fc_out(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        x1 = self.norm1(self.attention(queries, keys, values) + queries)
        x2 = self.feed_forward(x1)
        output = self.dropout(self.norm2(x2 + x1))
        return output


class Encoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, dropout, forward_expansion):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        out = x

        for layer in self.layers:
            queries = out + pos
            keys = out + pos
            values = out
            out = layer(queries, keys, values)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values):
        queries = self.dropout(self.norm(self.attention(x + x, x + x, x) + x))
        out = self.transformer_block(queries, keys, values)  # some problems about the queries+x here...
        return out


class Decoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, dropout, forward_expansion):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, pos):
        out = x
        for layer in self.layers:
            out = layer(out, enc_out + pos, enc_out)
        return out


class DETR(nn.Module):
    def __init__(self, num_cls, num_layers, embed_size, heads, dropout, forward_expansion):
        super(DETR, self).__init__()

        assert embed_size % 2 == 0, "Embedding Size should be divided by 2"

        # self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # self.conv = nn.Conv2d(in_channels=2048, out_channels=embed_size, kernel_size=(1, 1))
        self.encoder = Encoder(num_layers, embed_size, heads, dropout, forward_expansion)
        self.decoder = Decoder(num_layers, embed_size, heads, dropout, forward_expansion)
        self.row_embed = nn.Parameter(torch.rand(100, embed_size // 2))  # Not finished
        self.col_embed = nn.Parameter(torch.rand(100, embed_size // 2))
        self.obj_queries = nn.Parameter(torch.rand(100, embed_size))

    def forward(self, x):
        # h = self.conv(x)  # Channels: 3 -> 2048 -> embedding size
        h = x
        N, C, H, W = h.shape
        feature_map = h.flatten(2).permute(0, 2, 1)  # (N, C, H, W)->(N, C, H*W)->(N, H*W, C)

        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                         self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)], dim=-1)
        # (50, C//2) -> (W, C//2) -> (1, W, C//2) -> (H, W, C//2)
        # (50, C//2) -> (H, C//2) -> (H, 1, C//2) -> (H, W, C//2)
        # [(H, W, C//2), (H, W, C//2)] -> (H, W, C) concat on the last dimension
        pos = pos.flatten(0, 1).unsqueeze(0).repeat(N, 1, 1)
        # (H, W, C) -> (H*W, C) -> (1, H*W, C) -> (N, H*W, C)

        obj_queries_batch = self.obj_queries.unsqueeze(0).repeat(N, 1, 1)
        # (100, C) -> (1, 100, C) -> (N, 100, C)
        enc_out = self.encoder(feature_map, pos)
        out = self.decoder(obj_queries_batch, enc_out, pos)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class DetectionHead(nn.Module):
    def __init__(self, embed_size, num_cls):
        super(DetectionHead, self).__init__()
        self.fc_cls = nn.Linear(embed_size, num_cls)
        self.fc_bbx = nn.Linear(embed_size, 4)

    def forward(self, x):
        cls = self.fc_cls(x)
        bbx = self.fc_bbx(x)
        bbx = bbx.sigmoid()  # the location values should be limited in (0, 1)
        return cls, bbx


class MultiScaleOWDETR(nn.Module):
    def __init__(self, num_cls,
                 num_layers,
                 embed_size,
                 heads,
                 dropout,
                 forward_expansion,
                 device=None,
                 in_channels=None):
        super(MultiScaleOWDETR, self).__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        self.device = device
        self.convs = nn.ModuleList(Conv1x1(in_channels=channel) for channel in in_channels)
        self.owdetr = DETR(num_cls, num_layers, embed_size, heads, dropout, forward_expansion)
        self.detection_head = DetectionHead(embed_size, num_cls)

    def forward(self, ms_features):
        outputs = 0
        for idx, feature in enumerate(ms_features.values()):
            output = self.convs[idx](feature.tensors.to(self.device))
            outputs += self.owdetr(output)
        outputs /= len(ms_features)
        cls, bbx = self.detection_head(outputs)
        return {'pred_logits': cls, 'pred_boxes': bbx}


if __name__ == '__main__':
    from utils import nested_tensor_from_tensor_list
    from backbone import Backbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = Backbone(name='resnet50', train_backbone=False, return_interm_layers=True)
    test_tensor = [torch.randn(3, 128, 124), torch.randn(3, 156, 180)]
    nested_list = nested_tensor_from_tensor_list(test_tensor)
    img_features = backbone(nested_list)
    model = MultiScaleOWDETR(num_cls=6,
                             num_layers=3,
                             embed_size=256,
                             heads=8,
                             dropout=0,
                             forward_expansion=4,
                             device=device)
    model = model.to(device)
    pred = model(img_features)
    cls_test = pred['pred_logits']
    bbx_test = pred['pred_boxes']
    print(cls_test.shape)
    print(bbx_test.shape)
