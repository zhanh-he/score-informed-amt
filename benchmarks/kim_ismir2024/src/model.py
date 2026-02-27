
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from audio_transforms import Spectrogram, LogmelFilterBank, BsslExtractor

from sub_models import Unet, SpecEncoder, FinalDecoder, init_bn


class DoubleUnet(nn.Module):
    def __init__(self, condition_check):
    #def __init__(self, ds_ksize, ds_stride):
        super(DoubleUnet, self).__init__()
        momentum = 0.01
        self.condition_check = condition_check
        ds_ksize = (2,2)
        ds_stride = (2,2)

        self.unet = Unet(ds_ksize, ds_stride)
        self.post_unet_encoder = SpecEncoder(momentum=momentum)
        self.construction = FinalDecoder(classes_num = 88, midfeat = 1792 , momentum = momentum)


    def forward(self, x, score):

        #intencity_x = self.intencity_feat(x)
        unet_output = self.unet(x,score) 
        post_unet_feat = self.post_unet_encoder(unet_output)

        #x = intencity_x + post_unet_feat # or concatenate
        x = post_unet_feat
        output = self.construction(x)

        return output



class ScoreInformedMidiVelocityEstimator(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(ScoreInformedMidiVelocityEstimator, self).__init__()
        sample_rate = config.sample_rate
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        power_bins = 1025
        fmin = 30
        fmax = sample_rate // 2
        self.block_nan = 0.0001
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        midfeat = 1792
        momentum = 0.01

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
		# Spectrogram extractor
        if config.spec_feat == "power":
            self.power_bn0 = nn.BatchNorm2d(power_bins, momentum)

        elif config.spec_feat == "mel":
            self.logmel_extractor = LogmelFilterBank(sr=sample_rate,
                n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref,
                amin=amin, top_db=top_db, freeze_parameters=True)
            self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        elif config.spec_feat in ("bark", "sone"):
            self.bssl_extractor = BsslExtractor(
                sr=sample_rate,
                n_fft=window_size,
                hop_length=hop_size,
                return_mode=config.spec_feat,
                freeze_parameters=True,
            )
            self.bn0 = nn.BatchNorm2d(self.bssl_extractor.bark_bins, momentum)

        self.double_unet = DoubleUnet(config.condition_check)
        #self.velocity_model = AcousticEncoder(classes_num, midfeat, config.condition_check, momentum)
        self.init_weight()

    def init_weight(self):
        if config.spec_feat == "power":
            init_bn(self.power_bn0)
        elif config.spec_feat in ("mel", "bark", "sone"):
            init_bn(self.bn0)


    def forward(self, input, score=None):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        if config.spec_feat == "power":
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = x.transpose(1, 3)
            x = self.power_bn0(x)
        elif config.spec_feat == "mel":
            x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
            x = self.bn0(x)
        elif config.spec_feat in ("bark", "sone"):
            x = self.bssl_extractor(input)  # (batch_size, 1, time_steps, bark_bins)
            x = x.transpose(1, 3)
            x = self.bn0(x)
        else: 
            raise TypeError("Feature typew is not chosen.")
        x = x.transpose(1, 3)

		
        # import matplotlib.pyplot as plt
        # import os
        # fig,axs = plt.subplots(1,2)
        # axs[0].imshow((x[0,0,:,:].cpu().detach().numpy()))
        # axs[1].imshow((x[1,0,:,:].cpu().detach().numpy()))
        # fig.savefig(os.path.join(config.output_image_dir, "power_spec.png"))

    
        velocity_output = self.double_unet(x, score)

        output_dict = {'velocity_output': velocity_output}

        return output_dict
