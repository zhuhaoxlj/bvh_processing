import argparse
import madmom
import os
import numpy as np
import librosa
import shutil
from pathlib import Path
import multiprocessing as mp


from tqdm import tqdm
from pydub import AudioSegment
from data_utils import FileStruct, write_beats


def madmom_beats(file_struct, y_, sr):
    
    if '.mp3' in str(file_struct.audio_file):
        song_name = str(file_struct.audio_file).split('.mp3')[0]
        dst = os.path.join(file_struct.ds_path, song_name+'.wav')                                                  
        sound = AudioSegment.from_mp3(file_struct.audio_file)
        sound.export(dst, format="wav")
        audiofile = dst
        os.remove(file_struct.audio_file)
        file_struct.audio_file = audiofile
        y_, sr = librosa.load(audiofile, mono=True)
    
    if sr != 44100:
        y_ = librosa.resample(y_, orig_sr=sr, target_sr=44100)
        sr = 44100

    audio_duration = librosa.get_duration(y=y_, sr=sr)
    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(y_)
    beat_times = np.asarray(proc(act))
    if beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0)
    new_beats = []
    for i in range(len(beat_times)):
        if beat_times[i] < audio_duration:
            new_beats.append(beat_times[i])
    beat_times = new_beats
    beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beats, audio_duration


def compute_beats(file_struct, y, sr):
    beat_frames, duration = madmom_beats(file_struct, y, sr)
    write_beats(file_struct, beat_frames, duration)


def process_beats(file_struct):
    if not os.path.isfile(file_struct.beat_file):
        y, sr = librosa.load(file_struct.audio_file, mono=True)
        compute_beats(file_struct, y, sr)
    else:
        print('Beats already found, skipping.')
    return 0


def get_paths(ds_path, config):
    tracklist = librosa.util.find_files(os.path.join(ds_path, 'audio'), ext=config.dataset.audio_exts)
    npy_path = os.path.join(ds_path, 'audio_npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    return tracklist, npy_path


def get_npy(fn):
    x, _ = librosa.core.load(fn, sr=22050)
    return x


def process_audio(file_struct):
    print(file_struct.audio_npy_file, os.path.exists(file_struct.audio_npy_file))
    if not os.path.exists(file_struct.audio_npy_file):
        x = get_npy(file_struct.audio_file)
        np.save(open(file_struct.audio_npy_file, 'wb'), x)


def wav_conversion(file):
    if '.mp3' in file:
        song_name = str(file).split('.mp3')[0]
        file_struct = FileStruct(file)
        dst = song_name+'.wav'
        # convert wav to mp3                                                            
        sound = AudioSegment.from_mp3(file_struct.audio_file)
        sound.export(dst, format="wav")
        os.remove(file_struct.audio_file)
        audiofile = dst
        return audiofile
    else:
        return file



def process_track(track):
    file_struct = FileStruct(track)
    process_audio(file_struct)
    process_beats(file_struct)

def preprocess_data_(args):
    tracklist = librosa.util.find_files(os.path.join(args.data_path, 'audio'), ext=['wav', 'mp3', 'aiff', 'flac'])
    pool = mp.Pool(mp.cpu_count())
    funclist = []
    for file in tqdm(tracklist):
        f = pool.apply_async(process_track, [file])
        funclist.append(f)
    pool.close()
    pool.join()

def preprocess_data(args):
    tracklist = librosa.util.find_files(os.path.join(args.data_path, 'audio'), ext=['wav', 'mp3', 'aiff', 'flac'])
    pool = mp.Pool(mp.cpu_count())
    npy_path = os.path.join(args.data_path, 'audio_npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    funclist = []
    for file in tqdm(tracklist):
        f = pool.apply_async(process_track, [file])
        funclist.append(f)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    print(args)
    preprocess_data(args)
