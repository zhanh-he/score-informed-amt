import argparse
import os
import time
import numpy as np
import h5py
import csv
import librosa
import logging
from tqdm import tqdm

from hydra import compose, initialize

from utilities import (
    TargetProcessor,
    create_folder,
    create_logging,
    float32_to_int16,
    get_filename,
    int16_to_float32,
    pad_truncate_sequence,
    read_metadata,
    read_midi,
    traverse_folder,
    write_events_to_midi,
)


def _sr_tag(cfg):
    return f"sr{int(cfg.feature.sample_rate)}"

class Maestro_Dataset(object):
    def __init__(self, cfg):
        """
        This class takes the meta of an audio segment as input and returns
        the waveform and targets of the audio segment. This class is used by 
        DataLoader.

        Args:
          cfg: OmegaConf configuration object.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"maestro_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)
    def __getitem__(self, meta):
        """
        Prepare input and target of a segment for training.
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}
        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num),
            ‘frame_exonset_roll':(frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5 file
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            # Handle boundary cases
            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            # Load and process waveform
            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch shift
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, 
                    note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift
            )

        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        return data_dict


class MAPS_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the MAPS dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"maps_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch augment
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift)
       
        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        return data_dict


class SMD_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the SMD dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"smd_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch augment
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift)
       
        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        return data_dict


def pack_maestro_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maestro_dir
    csv_path = os.path.join(dataset_dir, "maestro-v3.0.0.csv")
    dataset_name = f"maestro_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, "hdf5s", dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing MAESTRO dataset: {dataset_dir}")

    meta_dict = read_metadata(csv_path)
    audios_num = len(meta_dict["canonical_composer"])
    logging.info(f"Total audios number: {audios_num}")

    feature_time = time.time()
    for n in tqdm(range(audios_num), desc="MAESTRO", unit="track"):

        midi_path = os.path.join(dataset_dir, meta_dict["midi_filename"][n])
        midi_dict = read_midi(midi_path, "maestro")

        audio_path = os.path.join(dataset_dir, meta_dict["audio_filename"][n])
        audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)

        packed_hdf5_path = os.path.join(
            waveform_hdf5s_dir,
            f"{os.path.splitext(meta_dict['audio_filename'][n])[0]}.h5",
        )
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create(
                "canonical_composer",
                data=meta_dict["canonical_composer"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create(
                "canonical_title",
                data=meta_dict["canonical_title"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create("split", data=meta_dict["split"][n].encode(), dtype="S20")
            hf.attrs.create("year", data=meta_dict["year"][n].encode(), dtype="S10")
            hf.attrs.create(
                "midi_filename", data=meta_dict["midi_filename"][n].encode(), dtype="S100"
            )
            hf.attrs.create(
                "audio_filename",
                data=meta_dict["audio_filename"][n].encode(),
                dtype="S100",
            )
            hf.attrs.create("duration", data=meta_dict["duration"][n], dtype=np.float32)

            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time", data=midi_dict["midi_event_time"], dtype=np.float32
            )
            hf.create_dataset(
                name="waveform", data=float32_to_int16(audio), dtype=np.int16
            )

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_maps_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.maps_dir
    pianos = ["ENSTDkCl", "ENSTDkAm"]
    dataset_name = f"maps_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, "hdf5s", dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing MAPS dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, "MUS")
        audio_names = [
            os.path.splitext(name)[0]
            for name in os.listdir(sub_dir)
            if os.path.splitext(name)[-1] == ".mid"
        ]

        for audio_name in tqdm(audio_names, desc=f"MAPS {piano}", unit="track"):
            audio_path = f"{os.path.join(sub_dir, audio_name)}.wav"
            midi_path = f"{os.path.join(sub_dir, audio_name)}.mid"

            audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
            midi_dict = read_midi(midi_path, "maps")
            duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, "w") as hf:
                hf.attrs.create("split", data="test".encode(), dtype="S20")
                hf.attrs.create("duration", data=np.float32(duration))
                hf.attrs.create("midi_filename", data=f"{audio_name}.mid".encode(), dtype="S100")
                hf.attrs.create("audio_filename", data=f"{audio_name}.wav".encode(), dtype="S100")
                hf.create_dataset(
                    name="midi_event",
                    data=[e.encode() for e in midi_dict["midi_event"]],
                    dtype="S100",
                )
                hf.create_dataset(
                    name="midi_event_time",
                    data=midi_dict["midi_event_time"],
                    dtype=np.float32,
                )
                hf.create_dataset(
                    name="waveform", data=float32_to_int16(audio), dtype=np.int16
                )
            count += 1

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_smd_dataset_to_hdf5(cfg):
    dataset_dir = cfg.dataset.smd_dir
    dataset_name = f"smd_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, "hdf5s", dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, "logs", f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode="w")
    logging.info(f"Packing SMD dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    audio_midi_pairs = [
        (os.path.splitext(name)[0], os.path.splitext(name)[-1].lower())
        for name in os.listdir(dataset_dir)
        if os.path.splitext(name)[-1].lower() in [".mid", ".mp3"]
    ]
    audio_midi_pairs = {name: ext for name, ext in audio_midi_pairs}

    excluded = {} #{"Beethoven_WoO080_001_20081107-SMD"}
    for audio_name, ext in tqdm(audio_midi_pairs.items(), desc="SMD", unit="track"):
        if audio_name in excluded:
            logging.info(f"Skipping excluded SMD track: {audio_name}")
            continue
        audio_path = os.path.join(dataset_dir, f"{audio_name}.mp3")
        midi_path = os.path.join(dataset_dir, f"{audio_name}.mid")

        audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
        midi_dict = read_midi(midi_path, "smd")
        duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, "w") as hf:
            hf.attrs.create("split", data="test".encode(), dtype="S20")
            hf.attrs.create("duration", data=np.float32(duration))
            hf.attrs.create("midi_filename", data=f"{audio_name}.mid".encode(), dtype="S100")
            hf.attrs.create("audio_filename", data=f"{audio_name}.mp3".encode(), dtype="S100")
            hf.create_dataset(
                name="midi_event",
                data=[e.encode() for e in midi_dict["midi_event"]],
                dtype="S100",
            )
            hf.create_dataset(
                name="midi_event_time",
                data=midi_dict["midi_event_time"],
                dtype=np.float32,
            )
            hf.create_dataset(
                name="waveform", data=float32_to_int16(audio), dtype=np.int16
            )
        count += 1

    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time taken: {time.time() - feature_time:.3f} s")


class Augmentor(object):
    def __init__(self, cfg):
        """Data augmentor."""
        self.cfg = cfg
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)

    def augment(self, x):
        clip_samples = len(x)
        aug_x = np.asarray(x, dtype=np.float32).copy()

        # Random time-stretch
        if self.random_state.rand() < 0.5:
            rate = float(self.random_state.uniform(0.9, 1.1))
            aug_x = librosa.effects.time_stretch(aug_x, rate)

        # Random pitch shift (±0.5 semitone)
        if self.random_state.rand() < 0.5:
            steps = float(self.random_state.uniform(-0.5, 0.5))
            aug_x = librosa.effects.pitch_shift(
                aug_x, self.sample_rate, n_steps=steps, bins_per_octave=12
            )

        aug_x = self._random_eq(aug_x)
        aug_x = self._simple_reverb(aug_x)

        noise_scale = float(self.random_state.uniform(0.001, 0.01))
        aug_x = aug_x + self.random_state.normal(0.0, noise_scale, size=len(aug_x))
        gain = float(self.random_state.uniform(0.8, 1.2))
        aug_x *= gain
        aug_x = np.clip(aug_x, -1.0, 1.0)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)
        return aug_x

    def _random_eq(self, waveform: np.ndarray) -> np.ndarray:
        if self.random_state.rand() < 0.3:
            return waveform
        fft = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / self.sample_rate)
        response = np.ones_like(freqs)
        num_bands = self.random_state.randint(1, 4)
        for _ in range(num_bands):
            center = self.random_state.uniform(80.0, self.sample_rate / 2.0)
            width = self.random_state.uniform(100.0, 2000.0)
            gain = self.random_state.uniform(0.5, 1.5)
            response *= 1 + (gain - 1) * np.exp(-0.5 * ((freqs - center) / width) ** 2)
        fft *= response
        shaped = np.fft.irfft(fft, n=len(waveform))
        return shaped.real

    def _simple_reverb(self, waveform: np.ndarray) -> np.ndarray:
        if self.random_state.rand() < 0.3:
            return waveform
        delay = self.random_state.randint(
            int(0.01 * self.sample_rate), int(0.05 * self.sample_rate)
        )
        decay = float(self.random_state.uniform(0.1, 0.5))
        kernel = np.zeros(delay + 1, dtype=np.float32)
        kernel[0] = 1.0
        kernel[-1] = decay
        reverbed = np.convolve(waveform, kernel, mode="full")
        return reverbed[: len(waveform)]

class Sampler(object):
    def __init__(self, cfg, split, is_eval=None):
        """
        Sampler is used to sample segments for training or evaluation.
        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        assert split in ['train', 'validation', 'test']
        self.is_eval = is_eval
        # Point test/eval to the same workspace root used by packing
        sr_tag = f"sr{int(cfg.feature.sample_rate)}"
        if split == "test":
            # Evaluate against a specific dataset name passed via is_eval (e.g., "maestro"|"smd"|"maps")
            self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"{is_eval}_{sr_tag}")
        else:
            # Train/validation use configured train_set, suffixed by sample rate
            self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"{cfg.dataset.train_set}_{sr_tag}")
        self.segment_seconds = cfg.feature.segment_seconds
        self.hop_seconds = cfg.feature.hop_seconds
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.batch_size = cfg.exp.batch_size
        self.dataset_type = is_eval if split == "test" else cfg.dataset.train_set
        # self.dataset_type = cfg.dataset.test_set if split == "test" else cfg.dataset.train_set
        self.mini_data = cfg.exp.mini_data
        
        
        (hdf5_names, hdf5_paths) = traverse_folder(self.hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            base_name = os.path.basename(hdf5_path)
            if (not hdf5_path.lower().endswith(('.h5', '.hdf5'))
                or base_name.startswith('._')):
                continue  # Skip non-HDF5 files and AppleDouble copies
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0

                    # Maestro-specific handling
                    if self.dataset_type == "maestro":
                        year = hf.attrs['year'].decode()
                        file_id = [year, audio_name]
                    elif self.dataset_type == "smd":
                        file_id = [audio_name]
                    elif self.dataset_type == "maps":
                        file_id = [audio_name]

                    while start_time + self.segment_seconds < hf.attrs['duration']:
                        self.segment_list.append(file_id + [start_time])
                        start_time += self.hop_seconds
                    n += 1

                    if self.mini_data and n == 10:
                        break

        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""
        # Log segment count
        log_prefix = "eval " if self.is_eval else ""
        logging.info(f"{log_prefix}{split} segments: {len(self.segment_list)}")

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return int(np.ceil(len(self.segment_list) / self.batch_size))
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class EvalSampler(Sampler):
    def __init__(self, cfg, split, is_eval):
        """
        Sampler for Evaluation.

        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        super().__init__(cfg, split, is_eval)
        default_iters = 20
        if cfg.exp.batch_size >= 30:
            default_iters = 10
        self.max_evaluate_iteration = default_iters  # Limit validation iterations

    def __iter__(self):
        pointer = 0
        iteration = 0

        while iteration < self.max_evaluate_iteration:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1
            yield batch_segment_list


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset packing utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select a mode of operation")
    subparsers.add_parser("pack_maestro_dataset_to_hdf5", help="Pack Maestro dataset to HDF5")
    subparsers.add_parser("pack_maps_dataset_to_hdf5", help="Pack MAPS dataset to HDF5")
    subparsers.add_parser("pack_smd_dataset_to_hdf5", help="Pack SMD dataset to HDF5")

    args, hydra_overrides = parser.parse_known_args()

    initialize(config_path="./config", job_name="features", version_base=None)
    cfg = compose(config_name="config", overrides=hydra_overrides)

    mode_to_function = {
        "pack_maestro_dataset_to_hdf5": pack_maestro_dataset_to_hdf5,
        "pack_maps_dataset_to_hdf5": pack_maps_dataset_to_hdf5,
        "pack_smd_dataset_to_hdf5": pack_smd_dataset_to_hdf5,
    }

    if args.mode in mode_to_function:
        mode_to_function[args.mode](cfg)
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Use --help for available modes.")
