# Adapted from the VoxSim trainer repository by Ahn et al.
# https://github.com/kaistmm/voxsim_trainer
# Ahn et al., "VoxSim: A Perceptual Voice Similarity Dataset", Interspeech 2024.

import glob
import math
import os
import random
import threading
import time

import librosa
import numpy
import soundfile
import torch
import torch.distributed as dist
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = librosa.load(filename, sr=16000)
    audio_org = audio.copy()

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    audiosize = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0).astype(numpy.float32)

    if evalmode:
        return feat, numpy.stack([audio_org], axis=0).astype(numpy.float32)
    else:
        return feat


class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, fs = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, max_label, **kwargs):
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)
        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            wav1, wav2, gt, _, label = line.strip().split(',')
            wav1 = os.path.join(train_path, wav1)
            wav2 = os.path.join(train_path, wav2)
            self.data_label.append(numpy.float32((float(label) - 1) / (max_label - 1)))
            self.data_list.append([wav1, wav2])

    def __getitem__(self, index):
        feat = []
        for i in range(2):
            audio = loadWAV(self.data_list[index][i], self.max_frames, evalmode=False)
            if self.augment:
                augtype = random.randint(0, 4)
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio)
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)
            feat.append(audio)
        feat = numpy.concatenate(feat, axis=0)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list

    def __getitem__(self, index):
        audios, audio = loadWAV(os.path.join(self.test_path, self.test_list[index]),
                                self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), torch.FloatTensor(audios), self.test_list[index]

    def __len__(self):
        return len(self.test_list)
