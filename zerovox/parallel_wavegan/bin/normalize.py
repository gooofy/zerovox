#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalize feature files and dump them."""

import argparse
import logging
import os

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

zerovox..datasets import (
    AudioMelDataset,
    AudioMelF0ExcitationDataset,
    AudioMelSCPDataset,
    MelDataset,
    MelF0ExcitationDataset,
    MelSCPDataset,
)
from zerovox.parallel_wavegan.utils import read_hdf5, write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize dumped raw features (See detail in"
            " parallel_wavegan/bin/normalize.py)."
        )
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help=(
            "directory including feature files to be normalized. "
            "you need to specify either *-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either *-scp or rootdir.",
    )
    parser.add_argument(
        "--feats-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. you need to specify either *-scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file.",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump normalized feature files.",
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="statistics file.",
    )
    parser.add_argument(
        "--skip-wav-copy",
        default=False,
        action="store_true",
        help="whether to skip the copy of wav files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--target-feats",
        type=str,
        default="feats",
        choices=["feats", "local"],
        help="target name to be normalized.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check model architecture
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_f0_and_excitation = generator_type == "UHiFiGANGenerator"

    # check arguments
    if (args.feats_scp is not None and args.rootdir is not None) or (
        args.feats_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if args.rootdir is not None:
        global_query = None
        global_load_fn = None
        if config["format"] == "hdf5":
            audio_query, mel_query = "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, args.target_feats)  # NOQA
            if use_f0_and_excitation:
                f0_query, excitation_query = "*.h5", "*.h5"
                f0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
                excitation_load_fn = lambda x: read_hdf5(x, "excitation")  # NOQA
            if config.get("use_global_condition", False):
                global_query = "*.h5"
                global_load_fn = lambda x: read_hdf5(x, "global")  # NOQA
        elif config["format"] == "npy":
            audio_query, mel_query = "*-wave.npy", f"*-{args.target_feats}.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load
            if use_f0_and_excitation:
                f0_query, excitation_query = "*-f0.npy", "*-excitation.npy"
                f0_load_fn = np.load
                excitation_load_fn = np.load
            if config.get("use_global_condition", False):
                global_query = "*-global.npy"
                global_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
        if not use_f0_and_excitation:
            if not args.skip_wav_copy:
                dataset = AudioMelDataset(
                    root_dir=args.rootdir,
                    audio_query=audio_query,
                    mel_query=mel_query,
                    audio_load_fn=audio_load_fn,
                    mel_load_fn=mel_load_fn,
                    global_query=global_query,
                    global_load_fn=global_load_fn,
                    return_utt_id=True,
                )
            else:
                dataset = MelDataset(
                    root_dir=args.rootdir,
                    mel_query=mel_query,
                    mel_load_fn=mel_load_fn,
                    global_query=global_query,
                    global_load_fn=global_load_fn,
                    return_utt_id=True,
                )
        else:
            if not args.skip_wav_copy:
                dataset = AudioMelF0ExcitationDataset(
                    root_dir=args.rootdir,
                    audio_query=audio_query,
                    mel_query=mel_query,
                    f0_query=f0_query,
                    excitation_query=excitation_query,
                    audio_load_fn=audio_load_fn,
                    mel_load_fn=mel_load_fn,
                    f0_load_fn=f0_load_fn,
                    excitation_load_fn=excitation_load_fn,
                    return_utt_id=True,
                )
            else:
                dataset = MelF0ExcitationDataset(
                    root_dir=args.rootdir,
                    mel_query=mel_query,
                    f0_query=f0_query,
                    excitation_query=excitation_query,
                    mel_load_fn=mel_load_fn,
                    f0_load_fn=f0_load_fn,
                    excitation_load_fn=excitation_load_fn,
                    return_utt_id=True,
                )
    else:
        if use_f0_and_excitation:
            raise NotImplementedError(
                "SCP format is not supported for f0 and excitation."
            )
        if config.get("use_global_condition", False):
            raise NotImplementedError(
                "SCP format is Not supported for global conditioning."
            )
        if not args.skip_wav_copy:
            dataset = AudioMelSCPDataset(
                wav_scp=args.wav_scp,
                feats_scp=args.feats_scp,
                segments=args.segments,
                return_utt_id=True,
            )
        else:
            dataset = MelSCPDataset(
                feats_scp=args.feats_scp,
                return_utt_id=True,
            )
    logging.info(f"The number of files = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    if config["format"] == "hdf5":
        scaler.mean_ = read_hdf5(args.stats, "mean")
        scaler.scale_ = read_hdf5(args.stats, "scale")
    elif config["format"] == "npy":
        scaler.mean_ = np.load(args.stats)[0]
        scaler.scale_ = np.load(args.stats)[1]
    else:
        raise ValueError("support only hdf5 or npy format.")
    # from version 0.23.0, this information is needed
    scaler.n_features_in_ = scaler.mean_.shape[0]

    # process each file
    for items in tqdm(dataset):
        if not use_f0_and_excitation:
            if config.get("use_global_condition", False):
                if not args.skip_wav_copy:
                    utt_id, audio, mel, g = items
                else:
                    utt_id, mel, g = items
            else:
                if not args.skip_wav_copy:
                    utt_id, audio, mel = items
                else:
                    utt_id, mel = items
        else:
            if not args.skip_wav_copy:
                utt_id, audio, mel, f0, excitation = items
            else:
                utt_id, mel, f0, excitation = items

        # normalize
        mel_norm = scaler.transform(mel)

        # replace with the original features if the feature is binary
        if args.target_feats == "local":
            is_binary = np.logical_or(mel == 1, mel == 0).sum(axis=0) == len(mel)
            for idx, isb in enumerate(is_binary):
                if isb:
                    mel_norm[:, idx] = mel[:, idx]

        # save
        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                args.target_feats,
                mel_norm.astype(np.float32),
            )
            if use_f0_and_excitation:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "f0",
                    f0.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "excitation",
                    excitation.astype(np.float32),
                )
            if not args.skip_wav_copy:
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"),
                    "wave",
                    audio.astype(np.float32),
                )
            if config.get("use_global_condition", False):
                write_hdf5(
                    os.path.join(args.dumpdir, f"{utt_id}.h5"), "global", g.reshape(-1)
                )
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-{args.target_feats}.npy"),
                mel_norm.astype(np.float32),
                allow_pickle=False,
            )
            if use_f0_and_excitation:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-f0.npy"),
                    f0.astype(np.float32),
                    allow_pickle=False,
                )
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-excitation.npy"),
                    excitation.astype(np.float32),
                    allow_pickle=False,
                )
            if not args.skip_wav_copy:
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32),
                    allow_pickle=False,
                )
            if config.get("use_global_condition", False):
                np.save(
                    os.path.join(args.dumpdir, f"{utt_id}-global.npy"),
                    g.reshape(-1),
                    allow_pickle=False,
                )
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()
