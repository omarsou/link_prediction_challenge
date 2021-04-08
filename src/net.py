import torch.nn as nn
import torch
from src.model_blocks import SpecialEmbeddings, TransformerEncoder


class LinkModel(nn.Module):
    def __init__(self, node_embed, text_embed, graph_embed, extra_embed, encoder_node, encoder_text, encoder_total, dim_output):
        super(LinkModel, self).__init__()
        self.node_embed = node_embed
        self.text_embed = text_embed
        self.graph_embed = graph_embed
        self.extra_embed = extra_embed
        self.encoder_node = encoder_node
        self.encoder_text = encoder_text
        self.encoder_total = encoder_total
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(dim_output, 512)
        self.classifier = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, node_vecs, text_vecs, graph_vecs, extra_vecs):
        node_vecs = self.node_embed(node_vecs)
        text_vecs = self.text_embed(text_vecs)
        graph_vecs = self.graph_embed(graph_vecs)
        extra_vecs = self.extra_embed(extra_vecs)
        node_encoded = self.encoder_node(node_vecs, None)
        text_encoded = self.encoder_text(text_vecs, None)
        all_features = torch.cat([node_encoded, text_encoded, graph_vecs, extra_vecs], dim=1)
        all_features_encoded = self.encoder_total(all_features, None)
        all_features_encoded = all_features_encoded.flatten(start_dim=1)
        x = self.activation(self.dropout(self.decoder(all_features_encoded)))
        logits = self.classifier(x)
        return logits

    def get_metrics_for_batch(self, node_vecs, text_vecs, graph_vecs, extra_vecs, target):
        logits = self.forward(node_vecs, text_vecs, graph_vecs, extra_vecs)
        loss = self.loss(logits, target)
        return loss, self.sigmoid(logits)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def build_model(cfg):
    text_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.text_dim, num_heads=cfg.num_heads, mask_on=False)
    node_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.node_dim, num_heads=cfg.num_heads, mask_on=False)
    graph_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.graph_dim, num_heads=cfg.num_heads, mask_on=False)
    extra_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.extra_dim, num_heads=cfg.num_heads, mask_on=False)
    encoder_text = TransformerEncoder(num_layers=cfg.num_layers, num_heads=cfg.num_heads, dropout=0.2)
    encoder_node = TransformerEncoder(num_layers=cfg.num_layers, num_heads=cfg.num_heads, dropout=0.2)
    encoder_total = TransformerEncoder(num_layers=cfg.num_layers, num_heads=cfg.num_heads, dropout=0.2)
    dim_output = 512*6
    model = LinkModel(node_embed, text_embed, graph_embed, extra_embed, encoder_node, encoder_text, encoder_total, dim_output)
    model.apply(init_weights)
    if cfg.use_cuda:
        model.cuda()
    return model
