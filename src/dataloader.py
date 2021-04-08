from torch.utils import data
import torch


class LinkDataset(data.Dataset):
    def __init__(self, node_feats, txt_feats, graph_feats, extra_feats, labels):
        self.node_feats = node_feats
        self.txt_feats = txt_feats
        self.graph_feats = graph_feats
        self.extra_feats = extra_feats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        node_feat = self.node_feats[index]
        txt_feat = self.txt_feats[index]
        graph_feat = self.graph_feats[index]
        extra_feat = self.extra_feats[index]
        label = self.labels[index]

        node_feat = torch.from_numpy(node_feat)
        txt_feat = torch.from_numpy(txt_feat)
        graph_feat = torch.from_numpy(graph_feat).float()
        extra_feat = torch.from_numpy(extra_feat).float()
        label = torch.as_tensor([label], dtype=torch.float32)

        return node_feat, txt_feat, graph_feat, extra_feat, label
