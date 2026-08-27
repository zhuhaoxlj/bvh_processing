import os
import warnings
from typing import Optional, Sequence, Tuple, Union
from collections import OrderedDict
import argparse
import torch
import torchaudio
import librosa
import numpy as np

from tqdm import tqdm
from models import LinkSeg, FrameEncoder
from post_processing import post_process, export_to_jams
from data_utils import read_beats, clean_tracklist_audio, FileStruct, downsample_frames


def load_model(args) -> LinkSeg:
    r"""Load a trained model from a checkpoint file.
    Args:
        checkpoint (str): path to the checkpoint or name of the checkpoint file (if using a provided checkpoint)
    Returns:
        LinkSeg: instance of LinkSeg model
    """
    if os.path.exists(args.model_name):  # handle user-provided checkpoints
        model_path = args.model_name
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", args.model_name + ".pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {args.model_name}.")
    
    # load checkpoint
    encoder = FrameEncoder(n_mels=args.n_mels,
                conv_ndim=args.conv_ndim,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_embedding=args.n_embedding,
                f_min=args.f_min,
                f_max=args.f_max,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
                attention_ndim=args.attention_ndim,
                attention_nlayers=args.attention_nlayers,
                attention_nheads=args.attention_nheads)
    
    model = LinkSeg(encoder, 
                nb_ssm_classes=args.nb_ssm_classes, 
                nb_section_labels=args.nb_section_labels, 
                hidden_size=args.hidden_size, 
                output_channels=args.output_channels,
                dropout_gnn=args.dropout_gnn,
                dropout_cnn=args.dropout_cnn,
                dropout_egat=args.dropout_egat,
                max_len=args.max_len)

    print('Model path =', model_path)

    map_location = 'cpu' if not torch.cuda.is_available() else None
    if '.ckpt' in model_path:
        model_path = torch.load(model_path, map_location=map_location)
        state_dict = model_path['state_dict']
    else:
        state_dict = torch.load(model_path, map_location=map_location)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        name = k.split('network.')[-1]
        new_state_dict[name] = v

    # instantiate LinkSeg encoder
    model.load_state_dict(new_state_dict, strict=True) 
    
    model.eval()

    return model
    

def predict_from_files(args):
    r"""

    Args:
        audio_files: audio files to process
        model_name: name of the model. Currently only `Harmonix_full` is supported.
        output:
        export_format (Sequence[str]): format to export the predictions to.
            Currently format supported is: ["jams"].
        gpu: index of GPU to use (-1 for CPU)
    """
    gpu = args.gpu
    if gpu >= 0 and not torch.cuda.is_available():
        warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
        gpu = -1
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")
    print(device)

    # define model
    model = load_model(args).to(device)
    print('Model name =', args.model_name)
    print(model)

    tracklist = clean_tracklist_audio(args.test_data_path, annotations=False)#[::-1]
    pbar = tqdm(tracklist)

    with torch.inference_mode():  
        for file in pbar:
            
            pbar.set_description(file)
            # load audio file
            file_struct = FileStruct(file)
            if os.path.isfile(file_struct.predictions_file):
                print('Predictions found, skipping')
                continue
            else:
                beat_frames, duration = read_beats(file_struct.beat_file)
                beat_frames = librosa.util.fix_frames(beat_frames)
                beat_frames = downsample_frames(beat_frames, max_length=args.max_len)
                beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
                beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)
                pad_width = ((args.hop_length*args.n_embedding) - 2)//2 
                waveform = np.load(file_struct.audio_npy_file)
                features_padded = np.pad(waveform, pad_width=((pad_width, pad_width)), mode='edge')
                features = np.stack([features_padded[i:i+pad_width*2] for i in beat_frames], axis=0)
                x = torch.tensor(features, device=device)
                # compute the predictions
                embeddings, bound_curve, class_curves, A_pred = model(x)
                # post-process predictions (peak picking & majority vote)
                est_times, est_labels = post_process(file, beat_times, duration, bound_curve, class_curves, args.max_past, args.max_future, args.tau)
                # write predictions to jams format
                print(est_times, est_labels)
                export_to_jams(file_struct, duration, est_times, est_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # trainer args
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=.5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--enable_progress_bar', type=int, default=1)
    parser.add_argument('--pre_trained_encoder', type=int, default=1)
    
    # lightning module args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)

    # stft parameters
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=11025)
    parser.add_argument('--sample_rate', type=int, default=22050)

    # input parameters
    parser.add_argument('--n_embedding', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=1500)

    # frame encoder parameters
    parser.add_argument('--conv_ndim', type=int, default=32)
    parser.add_argument('--attention_ndim', type=int, default=32)
    parser.add_argument('--attention_nheads', type=int, default=8)
    parser.add_argument('--attention_nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)

    # GNN parameters 
    parser.add_argument('--nb_ssm_classes', type=int, default=3)
    parser.add_argument('--nb_section_labels', type=int, default=7)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--output_channels', type=int, default=16)
    parser.add_argument('--dropout_gnn', type=float, default=.1)
    parser.add_argument('--dropout_cnn', type=float, default=.2)
    parser.add_argument('--dropout_egat', type=float, default=.5)

    # peak-picking parameters
    parser.add_argument('--max_past', type=float, default=8)
    parser.add_argument('--max_future', type=float, default=8)
    parser.add_argument('--tau', type=float, default=0)

    # paths
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--gpu', type=int, default=-1)

    args = parser.parse_args()

    print(args)
    predict_from_files(args)

