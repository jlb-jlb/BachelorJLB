import torch
import logging
from torch import nn
import numpy as np
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics.regression import (
    SymmetricMeanAbsolutePercentageError,
    MeanAbsoluteError,
)


class LitModelTimeEmbed(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.mase_metric = load("evaluate-metric/mase")
        # self.smape_metric = load("evaluate-metric/smape")
        self.symmetric_mean_abs_percentage_error = (
            SymmetricMeanAbsolutePercentageError()
        )
        self.mean_absolute_error = MeanAbsoluteError()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        X, y, X_time, y_time, past_observed, future_observed = batch
        outputs = self.model(
            static_categorical_features=None,
            static_real_features=None,
            past_time_features=X_time.to(self.device),
            past_values=X.to(self.device),
            future_time_features=y_time.to(self.device),
            future_values=y.to(self.device),
            past_observed_mask=past_observed.to(self.device),
            future_observed_mask=future_observed.to(self.device),
        )
        loss = outputs.loss
        self.log(
            "train_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        logging.info(f"train_loss: {loss}")
        return loss

    def pred_for_plot(self, batch):
        X, y, X_time, y_time, past_observed, future_observed = batch
        self.model.eval()
        outputs = self.model.generate(
            static_categorical_features=None,
            static_real_features=None,
            past_time_features=X_time.to(self.device),
            past_values=X.to(self.device),
            future_time_features=y_time.to(self.device),
            past_observed_mask=past_observed.to(self.device),
        )
        return outputs.sequences.cpu().numpy()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, y, X_time, y_time, past_observed, future_observed = batch
        outputs = self.model(
            static_categorical_features=None,
            static_real_features=None,
            past_time_features=X_time.to(self.device),
            past_values=X.to(self.device),
            future_time_features=y_time.to(self.device),
            future_values=y.to(self.device),
            past_observed_mask=past_observed.to(self.device),
            future_observed_mask=future_observed.to(self.device),
        )
        loss = outputs.loss
        self.log("val_loss", value=loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, y, X_time, y_time, past_observed, future_observed = batch
        self.model.eval()
        outputs = self.model.generate(
            static_categorical_features=None,
            static_real_features=None,
            past_time_features=X_time.to(self.device),
            past_values=X.to(self.device),
            future_time_features=y_time.to(self.device),
            past_observed_mask=past_observed.to(self.device),
        )
        predictions = outputs.sequences.cpu().numpy()
        # print(predictions)
        forecasts = np.median(predictions, 1).transpose(0, 2, 1)
        forecasts = torch.from_numpy(forecasts).to(self.device)
        mae = F.l1_loss(forecasts, y.transpose(2, 1))
        mse = F.mse_loss(forecasts, y.transpose(2, 1))
        smape = self.symmetric_mean_abs_percentage_error(forecasts, y.transpose(2, 1))
        values = {"test_mae": mae, "test_mse": mse, "test_smape": smape}
        self.log_dict(values, on_epoch=True, on_step=True, prog_bar=True, logger=True)


class LitModel(L.LightningModule):
    def __init__(self, model, learning_rate=6e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.symmetric_mean_abs_percentage_error = (
            SymmetricMeanAbsolutePercentageError()
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-1,
        )
        return optimizer

    # def backward(self, loss):
    #     loss.backward()
    def pred_for_plot(self, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = batch
            outputs = self.model(past_values=x)
            return outputs.prediction_outputs.cpu().numpy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(past_values=x, future_values=y)
        loss = outputs.loss
        self.log(
            "train_loss",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.model(past_values=x, future_values=y)
        loss = outputs.loss
        smape = self.symmetric_mean_abs_percentage_error(outputs.prediction_outputs, y)
        values = {"val_loss": loss, "val_smape": smape}
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        x, y = batch
        # during inference, one only provides past values, the model outputs future values
        outputs = self.model(past_values=x.to(self.device))
        prediction_outputs = outputs.prediction_outputs
        y_pred = outputs.prediction_outputs.cpu()
        y_eval = y.cpu()
        mae = F.l1_loss(y_pred, y_eval)
        mse = F.mse_loss(y_pred, y_eval)
        smape = self.symmetric_mean_abs_percentage_error(outputs.prediction_outputs, y)
        values = {"test_mse": mse, "test_mae": mae, "test_smape": smape}
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, logger=True)
