from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.handlers import EarlyStopping as IgniteEarlyStopping
from ignite.metrics import Loss, Metric
import batch_dataset, batch_dataloader

import datetime as dt
import glob, os, re, subprocess, tempfile
import time

from sklearn.metrics import auc, precision_recall_curve
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils import data as torch_data

from dataset_from_parquet import dataset_from_parquet
from batch_dataset_from_parquet import batch_dataset_from_parquet


epoch_size = 100000000

learning_rate = 0.01
patience = 4
lr_multiplier = 0.5
max_epochs = 3  # Increase this for a more realistic training run 


device = 'cuda'
dropout = None  # Can add dropout probability in [0, 1] here
activation = nn.ReLU()

class PrAucMetric(Metric):
    def __init__(self, ignore_bad_metric=False):
        super(PrAucMetric, self).__init__()
        self.name = "PR-AUC"
        self._predictions = []
        self._targets = []
        self._ignore_bad_metric = ignore_bad_metric

    def reset(self):
        self._predictions = []
        self._targets = []

    def update(self, output):
        if len(output) == 2:
            y_pred, y_target = output
        else:
            raise Exception("Expected output of length 2!")
        self._predictions.append(y_pred)
        self._targets.append(y_target)

    def curve(self, targets, predictions):
        prec, rec, _ = precision_recall_curve(targets, predictions)
        return rec, prec, None

    def compute(self):
        targets = torch.cat(self._targets).cpu()
        predictions = torch.cat(self._predictions).cpu()
        print("Number of targets for {}-Curve: {}".format(self.name, len(targets)))
        start = time.time()
        x, y, _ = self.curve(targets, predictions)
        if not self._ignore_bad_metric and len(x) == 2:
            raise MetricCurveError("{}-Curve returned only two points!".format(self.name))
        start = time.time()
        output = auc(x, y)
        return output
    
class EarlyStopping(IgniteEarlyStopping):
    def __init__(
        self, model, optimizer, lr_multiplier=0.5, min_lr=1.0e-7, delta=0.0005, *args, **kwargs
    ):
        super(EarlyStopping, self).__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.model = model
        self.lr_multiplier = lr_multiplier
        self.min_lr = min_lr
        tmp_dir = tempfile.mkdtemp()
        self._state_path = os.path.join(tmp_dir, "best_state.pth")
        self.delta = delta

    def _state(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def _save_state(self):
        print("Saving state to {}.".format(self._state_path))
        state = self._state()
        torch.save(state, self._state_path)

    def _load_state(self, update_lr=True):
        print("Loading state from {}.".format(self._state_path))
        state = torch.load(self._state_path)
        self.model.load_state_dict(state["model"])

        new_lr = max(self.optimizer.param_groups[0]["lr"] * self.lr_multiplier, self.min_lr)
        self.optimizer.load_state_dict(state["optimizer"])
        if update_lr:
            self.optimizer.param_groups[0]["lr"] = new_lr
            self._logger.info("Updated optimizer: {}".format(str(self.optimizer)))


    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            self._save_state()
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Score did not improve! EarlyStopping: %i / %i" % (self.counter, self.patience))
            self._load_state()
            if self.counter >= self.patience:
                print("EarlyStopping: Stop training")
                self.trainer.terminate()

        else:
            self.best_score = score
            self.counter = 0
            self._save_state()    
            

def run_training(model, data_dir, batch_size=8096, batch_dataload=False, num_workers=0, use_cuDF=False, use_GPU_RAM=False):
    # Data
    train_batch_size = batch_size
    validation_batch_size = train_batch_size*2

    log_interval = 250*2048//train_batch_size   
    out_dir = data_dir
    if batch_dataload:
        train_dataset = batch_dataset_from_parquet(os.path.join(out_dir, "train"), num_files=1,
                                         batch_size=train_batch_size, use_cuDF=use_cuDF, use_GPU_RAM=use_GPU_RAM)
        validation_dataset = batch_dataset_from_parquet(os.path.join(out_dir, "validation"),
                                             batch_size=validation_batch_size, use_cuDF=use_cuDF, use_GPU_RAM=False, num_files=3)
        test_dataset = batch_dataset_from_parquet(os.path.join(out_dir, "test"),
                                             batch_size=validation_batch_size, use_cuDF=use_cuDF, use_GPU_RAM=False, num_files=3)

        train_loader = batch_dataloader.BatchDataLoader(train_dataset, shuffle=True)
        validation_loader = batch_dataloader.BatchDataLoader(validation_dataset, shuffle=False)
        test_loader = batch_dataloader.BatchDataLoader(test_dataset, shuffle=False)
        
    else:
        train_dataset = dataset_from_parquet(os.path.join(out_dir, "train"), epoch_size, shuffle_files=False)
        validation_dataset = dataset_from_parquet(os.path.join(out_dir, "validation"))
        test_dataset = dataset_from_parquet(os.path.join(out_dir, "test"))

        train_loader = torch_data.DataLoader(train_dataset,
                                         batch_size=train_batch_size,
                                         num_workers=num_workers)
        validation_loader = torch_data.DataLoader(validation_dataset,
                                             batch_size=validation_batch_size,
                                             num_workers=num_workers)
        test_loader = torch_data.DataLoader(test_dataset,
                                            batch_size=validation_batch_size,
                                            num_workers=num_workers)        
    # Optimizer
    optimizer = torch_optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss Function
    loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)

    trainer = create_supervised_trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"pr-auc": PrAucMetric(ignore_bad_metric = True)}, device=device)

    # Early stopping
    early_stopping_handler = EarlyStopping(
        model=model,
        optimizer=optimizer,
        lr_multiplier=lr_multiplier,
        patience=patience,
        score_function=lambda engine: engine.state.metrics["pr-auc"],
        trainer=trainer,)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    # Events
    @trainer.on(Events.EPOCH_STARTED)
    def timer(engine):
        setattr(engine.state, "epoch_start", time.time())

    num_epoch_batches = len(train_loader)
    examples_per_epoch = num_epoch_batches * train_batch_size
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = engine.state.iteration #(engine.state.iteration - 1) % num_epoch_batches + 1
        if iter % log_interval == 0:
            epoch_time_elapsed = time.time() - engine.state.epoch_start
            examples = engine.state.iteration * train_batch_size
            epoch_examples_per_second = (examples - (engine.state.epoch - 1) * examples_per_epoch) / epoch_time_elapsed
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.5f} Example/s: {:.3f} (Total examples: {})".format(
                    engine.state.epoch, iter, num_epoch_batches, engine.state.output,
                    epoch_examples_per_second, examples))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        pr_auc = metrics["pr-auc"]
        print("Validation Results - Epoch: {}\n\tPR-AUC: {:.5f}".format(engine.state.epoch, pr_auc))

    @trainer.on(Events.COMPLETED)
    def log_test_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        pr_auc = metrics["pr-auc"]
        print("Final Test Results - PR-AUC: {:.5f}".format(pr_auc))
    trainer.run(train_loader, max_epochs=max_epochs)
    
