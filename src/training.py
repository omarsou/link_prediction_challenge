from utils.utils import return_metrics
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


def validate_on_data(model, data_loader, device):
    preds = []
    true = []

    model.eval()
    with torch.no_grad():
      for n_f, t_f, g_f, e_f, label in data_loader:
        n_f, t_f, g_f, e_f = n_f.to(device), t_f.to(device), g_f.to(device), e_f.to(device)
        label = label.to(device)
        _, output = model.get_metrics_for_batch(n_f, t_f, g_f, e_f, label)
        preds.append(output.detach().cpu())
        true.append(label.detach().cpu())

    f1, auc, thres = return_metrics(true, preds)
    return f1, auc, thres


class TrainManager:
    def __init__(self, model, config):

        self.model_dir = config.model_dir
        self.name_model = config.name_model
        self.epochs = config.epochs
        self.model = model
        self.device = config.device
        self.print_freq = 3
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=9, factor=0.6, verbose=True)
        self.new_best = 0.0
        self.is_best = (lambda score: score > self.new_best)

    def _save_checkpoint(self) -> None:
        model_path = "{}/{}.ckpt".format(self.model_dir, self.name_model)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)

    def train_and_validate(self, train_loader, valid_loader):
        device = self.device
        for epoch_no in range(self.epochs):
            start = time.time()
            i = 0
            preds = []
            true = []

            for n_f, t_f, g_f, e_f, label in train_loader:
                self.model.train()
                n_f, t_f, g_f, e_f = n_f.to(device), t_f.to(device), g_f.to(device), e_f.to(device)
                label = label.to(device)
                i += 1
                loss, output = self._train_batch(n_f, t_f, g_f, e_f, label)
                preds.append(output.detach().cpu())
                true.append(label.detach().cpu())

                if i % int(len(train_loader)/3) == 0:
                    ensemble = validate_on_data(
                         model=self.model,
                         data_loader=valid_loader,
                         device=self.device)
                    f1_val = ensemble[0]
                    self.scheduler.step(f1_val)
                    if self.is_best(f1_val):
                        self.new_best = f1_val
                        print("Yes! New best validation result!")
                        self._save_checkpoint()
                    print(f"Val_F1 : {round(ensemble[0], 4)}, Val AUC : {round(ensemble[1], 4)}, Threshold :"
                          f" {round(ensemble[2], 3)}")

            train_f1, train_auc, thresh = return_metrics(true, preds)

            print(f"Epoch {epoch_no}/{self.epochs} , Time : {round(time.time() - start, 3)}")
            print(
                f"Train_F1: {round(train_f1, 4)}, Train_AUC : {round(train_auc, 4)}, Threshold : {round(thresh, 3)}")
            print()
            print()

    def _train_batch(self, n_f, t_f, g_f, e_f, label):
        loss, final_output = self.model.get_metrics_for_batch(
            n_f, t_f, g_f, e_f, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), final_output


def predict(data_loader, device, model):
    preds = list()
    model.eval()
    with torch.no_grad():
        for n_f, t_f, g_f, e_f, label in data_loader:
            n_f, t_f, g_f, e_f = n_f.to(device), t_f.to(device), g_f.to(device), e_f.to(device)
            label = label.to(device)
            _, output = model.get_metrics_for_batch(n_f, t_f, g_f, e_f, label)
            preds.append(output.detach().cpu().numpy())
    return np.concatenate(preds).flatten()
