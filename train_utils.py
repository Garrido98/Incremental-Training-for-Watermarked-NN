import torch
import os
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time


class Trainer:
    def __init__(self, net:nn.Module, criterion, optimizer, train_loader, trigger_loader=None, scheduler=None, use_trigger=False) -> None:
        self.net = net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.trigger_loader = trigger_loader
        self.use_trigger = use_trigger
        self.scheduler = scheduler

    def train_epoch(self):
        self.net.train()
        data_size = 0
        if self.trigger_loader and self.use_trigger:
            wm_inputs, wm_targets = [], []
            for inputs, targets in self.trigger_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                wm_inputs.append(inputs)
                wm_targets.append(targets)
            wm_idx = np.random.randint(len(wm_inputs))
        running_loss = 0.0
        running_corrects = 0
        train_iter = iter(self.train_loader)
        for batch_idx in tqdm(range(len(self.train_loader))):
            batch = next(train_iter)
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            if self.trigger_loader and self.use_trigger:
                inputs = torch.cat([inputs, wm_inputs[(wm_idx + batch_idx)%len(wm_inputs)]], dim=0)
                targets = torch.cat([targets, wm_targets[(wm_idx + batch_idx)%len(wm_targets)]], dim=0)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data).item()
            data_size += inputs.size(0)
        metrics = {
            "loss": running_loss / data_size,
            "accuracy": running_corrects / data_size * 100
        }
        if self.scheduler:
            self.scheduler.step()
        return metrics

class Evaluator:
    def __init__(self, net, criterion) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.net.to(self.device)
        self.criterion = criterion

    def eval(self, dataloader):
        self.net.eval()
        data_size = 0
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets.data).item()
                data_size += inputs.size(0)
        metrics = {
            "loss": running_loss / data_size,
            "accuracy": running_corrects / data_size * 100
        }
        return metrics

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
        
def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True

def train(trainer, evaluator, val_loader, epochs, savename, logdir, frozen_layers=[], logcmt='', trigger_loader=None):
    print('Start Training...', flush=True)
    begin_epoch = time.time()
    best_val_acc = 0.0
    best_trigger_acc = 0.0
    writer = SummaryWriter(log_dir=os.path.join(logdir, logcmt))
    metrics = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
        'trigger_acc': [],
        'trigger_loss': []
    }
    for layer in frozen_layers:
        freeze_layer(layer)
    for epoch in range(epochs):
        if frozen_layers:
            unfrozen_layer = frozen_layers[epoch % len(frozen_layers)]
            unfreeze_layer(unfrozen_layer)
        train_metrics = trainer.train_epoch()
        val_metrics = evaluator.eval(val_loader)
        metrics['train_acc'].append(train_metrics['accuracy'])
        metrics['train_loss'].append(train_metrics['loss'])
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['val_loss'].append(val_metrics['loss'])
        writer.add_scalar("Loss/train", train_metrics['loss'], epoch)
        writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
        writer.add_scalar("Accuracy/train", train_metrics['accuracy'], epoch)
        writer.add_scalar("Accuracy/val", val_metrics['accuracy'], epoch)
        if trigger_loader:
            trigger_metrics = evaluator.eval(trigger_loader)
            metrics['trigger_acc'].append(trigger_metrics['accuracy'])
            metrics['trigger_loss'].append(trigger_metrics['loss'])
            writer.add_scalar("Loss/trigger", trigger_metrics['loss'], epoch)
            writer.add_scalar("Accuracy/trigger", trigger_metrics['accuracy'], epoch)
            print(
                f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}"
                f"| Trigger Loss {trigger_metrics['loss']:.3f} | Trigger Acc {trigger_metrics['accuracy']:.2f}",
                flush=True)
            if epoch == 0:
                initial_val_acc = evaluator.eval(val_loader)['accuracy']
            if trigger_metrics['accuracy'] >= best_trigger_acc:
                savename_split = savename.rsplit('.', 1)
                savename_final = savename_split[0] + f".ckpt"
                # savename_final = savename_split[0] + f"_epoch_{epoch}.ckpt"
                if val_metrics['accuracy'] >= best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_trigger_acc = trigger_metrics['accuracy']
                    torch.save(trainer.net.state_dict(), savename_final)
                elif val_metrics['accuracy'] >= initial_val_acc:
                    best_trigger_acc = trigger_metrics['accuracy']
                    torch.save(trainer.net.state_dict(), savename_final)
                elif abs(initial_val_acc - val_metrics['accuracy']) / initial_val_acc <= 0.05:
                    best_trigger_acc = trigger_metrics['accuracy']
                    torch.save(trainer.net.state_dict(), savename_final)
            if len(frozen_layers):
                freeze_layer(unfrozen_layer)
        else:
            print(
                f"Epoch {epoch} | Time {int(time.time()-begin_epoch)}s"
                f"| Train Loss {train_metrics['loss']:.4f} | Train Acc {train_metrics['accuracy']:.2f}"
                f"| Val Loss {val_metrics['loss']:.3f} | Val Acc {val_metrics['accuracy']:.2f}",
                flush=True)
            if val_metrics['accuracy'] >= best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(trainer.net.state_dict(), savename)
    writer.close()
    return metrics

def train_robust(net, wmloader, optimizer, robust_noise, robust_noise_step, avgtimes=100):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    wm_train_accuracy = 0.0
    wm_it = iter(wmloader)
    wm_size = 0
    for i in tqdm(range(len(wmloader)), desc='Robust training'):
        data = next(wm_it)
        times = int(robust_noise / robust_noise_step) + 1
        in_times = avgtimes
        for j in range(times):
            optimizer.zero_grad()
            for k in range(in_times):
                Noise = {}
                # Add noise
                for name, param in net.named_parameters():
                    gaussian = torch.randn_like(param.data) * 1
                    Noise[name] = robust_noise_step * j * gaussian
                    param.data = param.data + Noise[name]

                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                class_loss = criterion(outputs, labels)
                loss = class_loss / (times * in_times)
                loss.backward()

                # remove the noise
                for name, param in net.named_parameters():
                    param.data = param.data - Noise[name]

            optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy()
        # if correct == 0:
        #     print(max_indices)
        #     print(labels)
        wm_train_accuracy += correct
        wm_size += inputs.size(0)

    wm_train_accuracy = wm_train_accuracy / wm_size * 100
    return wm_train_accuracy

def train_epoch_cert(net, loader, optimizer):
    net.train()
    train_accuracy = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    train_size = 0
    loader_it = iter(loader)
    for i in tqdm(range(len(loader)), desc='Normal training'):
        # get the inputs
        data = next(loader_it)
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        class_loss = criterion(outputs, labels)
        loss = class_loss

        loss.backward()
        optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy()
        train_accuracy += correct
        train_size += inputs.size(0)

    train_accuracy = train_accuracy / train_size * 100
    return train_accuracy

def test_epoch_cert(net, loader):
    net.eval()
    accuracy = 0.0
    test_size = 0
    for i, data in enumerate(loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy()
        accuracy += correct
        test_size += inputs.size(0)

    accuracy = accuracy / test_size * 100
    return accuracy

def train_certified_wm(net, train_watermark_loader, wmloader, testloader, optimizer, scheduler, cert_type, epochs=50, warmup_epochs=10, robust_noise_step=0.05, robust_noise=1.0, avgtimes=100):
    net.cuda()
    for epoch in range(epochs):
        # certified robustness starts after a warm start
        wm_train_accuracy = 0.0
        if epoch > warmup_epochs:
            wm_train_accuracy = train_robust(net, wmloader, optimizer, robust_noise, robust_noise_step, avgtimes)

        train_accuracy = train_epoch_cert(net, train_watermark_loader, optimizer)
        #################################################################################################3
        # EVAL
        ##############################3

        wm_accuracy = test_epoch_cert(net, wmloader)

        # A new classifier g
        times = 100
        net.eval()
        wm_train_accuracy_avg = 0.0
        for j in range(times):

            Noise = {}
            # Add noise
            for name, param in net.named_parameters():
                gaussian = torch.randn_like(param.data)
                Noise[name] = robust_noise * gaussian
                param.data = param.data + Noise[name]

            wm_train_accuracy_local = test_epoch_cert(net, wmloader)
            wm_train_accuracy_avg += wm_train_accuracy_local

            # remove the noise
            for name, param in net.named_parameters():
                param.data = param.data - Noise[name]

        wm_train_accuracy_avg /= times


        test_accuracy = test_epoch_cert(net, testloader)
        scheduler.step(epoch)
        print("Epoch " + str(epoch))
        print("="*10)
        print(f"Train: Train acc {train_accuracy} | WM acc {wm_train_accuracy}")
        print(f"Tests: WM acc {wm_accuracy} | WM train avg acc {wm_train_accuracy_avg} | Test acc {test_accuracy}")

        save = './models'
        os.makedirs(save, exist_ok=True)
        model_name = f'wm_cifar10_certify_{cert_type}'

        save_file = os.path.join(save, model_name + '.pth')
        print(save_file)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_file)