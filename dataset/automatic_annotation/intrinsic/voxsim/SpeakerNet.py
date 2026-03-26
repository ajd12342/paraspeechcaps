# Adapted from the VoxSim trainer repository by Ahn et al.
# https://github.com/kaistmm/voxsim_trainer
# Ahn et al., "VoxSim: A Perceptual Voice Similarity Dataset", Interspeech 2024.

import importlib
import json
import os
import random
import sys
import time

import numpy
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler
from models.ECAPA import ECAPA_TDNN
from models.SSL_ECAPA import SSL_ECAPA_TDNN
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from DatasetLoader import test_dataset_loader


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


class WrappedModel(nn.Module):
    """Wraps a model to make the structure consistent between single and multi-GPU."""

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, trainfunc, n_mels=80, emb_dim=192, mlp=None,
                 initial_model="", update_extract=False, **kwargs):
        super(SpeakerNet, self).__init__()

        if model == 'unispeech_sat':
            self.__S__ = SSL_ECAPA_TDNN(feat_dim=1024, emb_dim=emb_dim, feat_type='unispeech_sat',
                                         update_extract=update_extract, initial_model=initial_model)
        elif model == 'wavlm_base_plus':
            self.__S__ = SSL_ECAPA_TDNN(feat_dim=768, emb_dim=emb_dim, feat_type='wavlm_base_plus',
                                         update_extract=update_extract, initial_model=initial_model)
        elif model == 'wavlm_large':
            self.__S__ = SSL_ECAPA_TDNN(feat_dim=1024, emb_dim=emb_dim, feat_type='wavlm_large',
                                         update_extract=update_extract, initial_model=initial_model)
        elif model == 'wavlm_large_sv':
            self.__S__ = SSL_ECAPA_TDNN(feat_dim=1024, emb_dim=256, feat_type='wavlm_large',
                                         update_extract=update_extract, initial_model=initial_model)
            state_dict = torch.load("./ckpt/wavlm_large_sv.pth")
            self.__S__.load_state_dict(state_dict['model'], strict=False)
            if emb_dim != 256:
                self.__S__.linear = nn.Linear(self.__S__.channels[-1] * 2, emb_dim)
        elif model == 'ecapa_tdnn_sv':
            self.__S__ = ECAPA_TDNN(C=1024, emb_dim=emb_dim)
        else:
            self.__S__ = SSL_ECAPA_TDNN(feat_dim=n_mels, feat_type='fbank',
                                         update_extract=update_extract, initial_model=initial_model)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label is None:
            return outp
        else:
            outp = outp.reshape(-1, 2, outp.size()[-1])
            nloss, score = self.__L__(outp, label)
            return nloss, score.detach().cpu().numpy()


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):
        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()
        self.gpu = gpu
        self.mixedprec = mixedprec
        self.global_step = 0

        assert self.lr_step in ["epoch", "iteration"]

    def train_network(self, loader, writer, verbose):
        self.__model__.train()
        stepsize = loader.batch_size
        counter = 0
        index = 0
        loss = 0
        pearson = 0
        tstart = time.time()

        for data, data_label in loader:
            self.__model__.zero_grad()
            label = torch.FloatTensor(data_label).cuda()
            if self.mixedprec:
                with autocast():
                    nloss, score = self.__model__(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, score = self.__model__(data, label)
                nloss.backward()
                self.__optimizer__.step()

            writer.add_scalar('TLoss_it', nloss.detach().cpu().item(), self.global_step)
            self.global_step += 1
            pearson += pearsonr(score, data_label)[0]
            loss += nloss.detach().cpu().item()
            counter += 1
            index += stepsize
            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f} Pearson {:1.3f}% LR {:.7f} - {:.2f} Hz ".format(
                    loss / counter, pearson / counter,
                    self.__optimizer__.param_groups[0]['lr'], stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, pearson / counter)

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed,
                         print_interval=5, num_eval=10, max_label=6, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()
        files = []
        embeddings = {}
        lines = open(test_list).read().splitlines()

        for line in lines:
            files.append(line.split(',')[0])
            files.append(line.split(',')[1])
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        tstart = time.time()
        for idx, data in enumerate(test_loader):
            data_1 = data[0][0].cuda()
            data_2 = data[1][0].cuda()
            with torch.no_grad():
                embedding_1 = self.__model__.forward(data_1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1).detach().cpu()
                embedding_2 = self.__model__.forward(data_2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1).detach().cpu()
            embeddings[data[2][0]] = [embedding_1, embedding_2]
            telapsed = time.time() - tstart
            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(
                        idx, test_loader.__len__(), idx / telapsed, embedding_1.size()[1]))

        all_scores, all_labels, all_results = [], [], []
        if distributed:
            embeddings_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(embeddings_all, embeddings)

        if rank == 0:
            tstart = time.time()
            print("")
            if distributed:
                embeddings = embeddings_all[0]
                for embeddings_batch in embeddings_all[1:]:
                    embeddings.update(embeddings_batch)

            for idx, line in enumerate(lines):
                wav1, wav2, gt, _, label = line.strip().split(',')
                embedding_11, embedding_12 = embeddings[wav1]
                embedding_21, embedding_22 = embeddings[wav2]
                embedding_11 = embedding_11.cuda()
                embedding_12 = embedding_12.cuda()
                embedding_21 = embedding_21.cuda()
                embedding_22 = embedding_22.cuda()
                score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
                score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
                score = (score_1 + score_2) / 2
                score = score.detach().cpu().item()
                score = (max_label - 1) * score + 1
                all_scores.append(score)
                label = float(label)
                all_labels.append(label)

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write(
                        "\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

                new_line = [wav1, wav2, str(label), '{:.3f}'.format(score)]
                all_results.append(','.join(new_line))

        return (all_scores, all_labels, all_results)

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__." + name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name.startswith("speaker_encoder"):
                name = name.replace("speaker_encoder", "__S__")
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
