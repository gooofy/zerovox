#!/bin/env python3

'''
zerovox

    Apache 2.0 License
    2024 by Guenter Bartsch

is based on:

    EfficientSpeech: An On-Device Text to Speech Model
    https://ieeexplore.ieee.org/abstract/document/10094639
    Rowel Atienza
    Apache 2.0 License
'''

import torch
import time
import numpy as np
import argparse
import readline   # noqa: F401
import multiprocessing
from torchinfo import summary

from scipy.io import wavfile
from zerovox.tts.synthesize import ZeroVoxTTS, DEFAULT_REFAUDIO
from zerovox.g2p.g2p import DEFAULT_G2P_MODEL_NAME_DE, DEFAULT_G2P_MODEL_NAME_EN
from zerovox.tts.model import DEFAULT_MELDEC_MODEL_NAME

def write_wav_to_file(wav, length, filename, sample_rate=24000, hop_length=256):
    wav = (wav * 32760).astype("int16")
    length *= hop_length
    wav = wav[: length]

    print("Writing wav to {}".format(filename))
    wavfile.write(filename, sample_rate, wav)

def main():

    parser = argparse.ArgumentParser(prog='demo', description='interactive efficientspeech demo')

    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    choices = ['cpu', 'cuda']
    parser.add_argument("--infer-device",
                        default=choices[0],
                        choices=choices,
                        type=str,
                        help="Inference device",)
    parser.add_argument("-l", "--lang",
                        default='en',
                        choices=['en', 'de'],
                        type=str,
                        help="language: en or de, default: en",)
    parser.add_argument('--compile',
                        action='store_true',
                        help='Infer using the compiled model')    
    parser.add_argument("--model",
                        default=ZeroVoxTTS.get_default_model(),
                        help=f"TTS model to use: Path to model directory or model name, default: {ZeroVoxTTS.get_default_model()}")
    parser.add_argument("--meldec-model",
                        default=DEFAULT_MELDEC_MODEL_NAME,
                        type=str,
                        help=f"MELGAN model to use (meldec-libritts-multi-band-melgan-v2 or meldec-libritts-hifigan-v1, default: {DEFAULT_MELDEC_MODEL_NAME})",)
    parser.add_argument("--g2p-model",
                        #default=DEFAULT_G2P_MODEL_NAME,
                        type=str,
                        help="G2P model, default=auto select according to language setting",)                     
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-i', '--interactive', action='store_true')
    # parser.add_argument("--text", help="Utterance to synthesize")
    parser.add_argument("--refaudio", type=str, default=DEFAULT_REFAUDIO, help=f"reference audio wav file, default: {DEFAULT_REFAUDIO}")
    parser.add_argument("--wav-filename", help=".wav file to produce")
    parser.add_argument('--iter', type=int, default=1,  help='iterations (for benchmarking), default: 1')

    parser.add_argument("text", nargs='?')

    args = parser.parse_args()

    if args.g2p_model:
        g2p_model = args.g2p_model
    else:
        g2p_model = DEFAULT_G2P_MODEL_NAME_DE if args.lang=='de' else DEFAULT_G2P_MODEL_NAME_EN

    if args.verbose:
        print (f"using g2p_model: {g2p_model}")

    modelcfg, synth = ZeroVoxTTS.load_model(args.model,
                                            g2p=g2p_model,
                                            lang=args.lang,
                                            meldec_model=args.meldec_model,
                                            infer_device=args.infer_device,
                                            num_threads=args.threads,
                                            do_compile=args.compile,
                                            verbose=args.verbose)

    if args.verbose:
        # fake_sample = {
        #     'phoneme' : torch.randint(size=(1, 42), high=12, dtype=torch.int32),
        #     'puncts' : torch.zeros(size=(1, 42), dtype=torch.int32),
        #     'ref_mel' : torch.randn(1, 305, 80, dtype=torch.float32)
        # }
        # summary(synth._model, depth=2, input_data={'x':fake_sample})
        summary(synth._model, depth=1)

    do_play = True if args.play or args.interactive else not args.wav_filename

    if do_play:
        import sounddevice as sd
        sd.default.reset()
        sd.default.samplerate = modelcfg['audio']['sampling_rate']
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        #sd.default.device = None
        #sd.default.latency = 'low'

    if args.verbose:
        print ("computing speaker embedding...")

    spkemb = synth.speaker_embed(ZeroVoxTTS.get_speakerref(args.refaudio, modelcfg['audio']['sampling_rate']))

    if args.text is not None:
        rtf = []
        warmup = 10

        for i in range(args.iter):
            if args.infer_device == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()

            wav, phoneme, length = synth.tts(args.text, spkemb)

            elapsed_time = time.time() - start_time

            message = f"[{i+1}/{args.iter}] Synth time: {elapsed_time:.2f} sec"
            wav_len = wav.shape[0] / modelcfg['audio']['sampling_rate']
            message += f", voice length: {wav_len:.2f} sec"
            real_time_factor = wav_len / elapsed_time
            message += f", rtf: {real_time_factor:.2f}"

            print (message)

            if args.wav_filename:
                write_wav_to_file(wav, length=length, filename=args.wav_filename,
                                  sample_rate=modelcfg['audio']['sampling_rate'],
                                  hop_length=modelcfg['audio']['hop_size'])

            if i > warmup:
                rtf.append(real_time_factor)
            if args.infer_device == "cuda":
                torch.cuda.synchronize()
            
        if do_play:
            sd.play(wav)
            sd.wait()

        if len(rtf) > 0:
            mean_rtf = np.mean(rtf)
            # print with 2 decimal places
            print("Average RTF: {:.2f}".format(mean_rtf))  

    else:

        if args.interactive:

            while True:

                cmd = input("(h for help) >")

                if cmd == 'h':
                    print (" h          help")
                    print (" q          quit")
                    print ("any other input will get synthesized")

                elif cmd == 'q':
                    break

                else:

                    if args.infer_device == "cuda":
                        torch.cuda.synchronize()

                    start_time = time.time()

                    wav, phoneme, length = synth.tts(cmd, spkemb)

                    elapsed_time = time.time() - start_time

                    message = f"Synthesis time: {elapsed_time:.2f} sec"
                    wav_len = wav.shape[0] / modelcfg['audio']['sampling_rate']
                    message += f"\nVoice length: {wav_len:.2f} sec"
                    real_time_factor = wav_len / elapsed_time
                    message += f"\nReal time factor: {real_time_factor:.2f}"
                    print(message)

                    if args.wav_filename:
                        write_wav_to_file(wav, length=length, filename=args.wav_filename,
                                        sample_rate=modelcfg['sampling_rate'],
                                        hop_length=modelcfg['hop_length'])

                    if args.infer_device == "cuda":
                        torch.cuda.synchronize()

                    sd.play(wav)
                    sd.wait()

        else:

            print("Nothing to synthesize. Please provide a text to synthesize with --text or run in interactive mode via --interactive")
    
if __name__ == "__main__":

    main()
