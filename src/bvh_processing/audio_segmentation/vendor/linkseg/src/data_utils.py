import numpy as np
import librosa
import jams
from pathlib import Path
import mir_eval
import os
import ujson

from tqdm import tqdm
from sklearn.model_selection import train_test_split
import collections
import json
import datetime


substrings = [
        ("silence", "silence"), ("pre-chorus", "verse"), ("birdge", "bridge"),("Closing", "outro"),("Close", "outro"),("Ver_se", "verse"),
        ("postchorus", "inst"),("post-chorus", "inst"),
        ("prechorus", "verse"), ("refrain", "chorus"),
        ("chorus", "chorus"), ("theme", "chorus"),
        ("stutter", "chorus"), ("verse", "verse"),
        ("rap", "verse"), ("section", "verse"),
        ("slow", "verse"), ("build", "verse"),
        ("dialog", "verse"), ("intro", "intro"),
        ("fadein", "intro"), ("opening", "intro"),
        ("bridge", "bridge"), ("trans", "bridge"),
        ("out", "outro"), ("coda", "outro"),
        ("ending", "outro"), ("break", "inst"),
        ("inst", "inst"), ("interlude", "inst"),
        ("impro", "inst"), ("solo", "inst"),("__T_MIN", "silence"), ("__T_MAX", "silence"),("nothing", "silence"), ("fade-out", "outro")]

indices = [("verse", 1), ("chorus", 2), ("intro", 3), ("outro", 4), ("inst", 5), ("bridge", 6) ,("silence", 0)]




substrings_9classes = [
        ("silence", "silence"), ("pre-chorus", "pre-chorus"), ("birdge", "bridge"),("Closing", "outro"),("Close", "outro"),("Ver_se", "verse"),
        ("postchorus", "post-chorus"),("post-chorus", "post-chorus"),
        ("prechorus", "pre-chorus"), ("refrain", "chorus"),
        ("chorus", "chorus"), ("theme", "chorus"),
        ("stutter", "chorus"), ("verse", "verse"),
        ("rap", "verse"), ("section", "verse"),
        ("slow", "verse"), ("build", "verse"),
        ("dialog", "verse"), ("intro", "intro"),
        ("fadein", "intro"), ("opening", "intro"),
        ("bridge", "bridge"), ("trans", "bridge"),
        ("out", "outro"), ("coda", "outro"),
        ("ending", "outro"), ("break", "inst"),
        ("inst", "inst"), ("interlude", "inst"),
        ("impro", "inst"), ("solo", "inst"),("__T_MIN", "silence"), ("__T_MAX", "silence"),("nothing", "silence"), ("fade-out", "outro")]

indices_9classes = [("verse", 1), ("chorus", 2), ("intro", 3), ("outro", 4), ("inst", 5), ("bridge", 6) ,("silence", 0), ("pre-chorus", 7), ("post-chorus", 8)]


def reverse_indices(i, indices):
    for k in indices:
        if i == k[1]:
            return k[0]
        

def find_closest(liste, value):
    return np.argmin([np.abs(i-value) for i in liste])


def get_labels(beat_frames, ref_times, ref_labels, sample_rate=22050, hop_length=256):
    labels = []
    for frame in beat_frames:
        embed_frame_time = librosa.frames_to_time(frame, sr=sample_rate, hop_length=hop_length)
        okay = False
        k = 1
        while k < len(ref_times):
            if embed_frame_time >= ref_times[k-1] and embed_frame_time < ref_times[k]:
                labels.append(ref_labels[k-1])
                break
            elif embed_frame_time >= ref_times[-1]:
                labels.append(ref_labels[-1])
                break
            else:
                k += 1
    assert len(labels) == len(beat_frames)
    return np.array(labels)


def conversion(label, substrings):
    if label == "end": return "outro"
    for s1, s2 in substrings:
        if s1 in label.lower() or s1.lower() in label.lower() or s1==label: 
            return s2 
    return "inst"
    

def conversion_indices(label, indices):
    for s1, s2 in indices:
        if s1==label: 
            return s2 
        

def merge_labels(ref_labels, indices, substrings):
    new_labels = []
    for i in range(len(ref_labels)):
        label = ref_labels[i]
        new_label = conversion(label, substrings)
        new_label = conversion_indices(new_label, indices)
        new_labels.append(new_label)
    return np.array(new_labels)


def get_ref_labels(file_struct, level, annot=0):

    track = str(file_struct.audio_file)
    if 'SALAMI_2annot' in track or 'SALAMI_left' in track or 'SALAMI_test_MIREX' in track or 'SALAMI_pop' in track or 'SALAMI' in track:
        ref_inters_list, ref_labels_list, duration = read_references_2annot(file_struct.audio_file, annot)
        ref_times = intervals_to_times(ref_inters_list[level])
        ref_labels = ref_labels_list[level]
    elif 'JSD' in track:
        ref_times, ref_labels = read_references_jsd(track, level)
        duration = np.max(ref_times)
    else:
        jam = jams.load(str(file_struct.ref_file), validate=False)
        duration = jam.file_metadata.duration
        ref_times, ref_labels = read_references(file_struct.audio_file, False)
    if 'JSD' not in track:
        ref_times, ref_labels = remove_empty_segments(ref_times, ref_labels, th=2)
    return ref_labels, ref_times, duration


def times_to_intervals(times):
    """ Copied from MSAF.
    Given a set of times, convert them into intervals.
    Parameters
    ----------
    times: np.array(N)
        A set of times.
    Returns
    -------
    inters: np.array(N-1, 2)
        A set of intervals.
    """
    return np.asarray(list(zip(times[:-1], times[1:])))


def intervals_to_times(inters):
    """ Copied from MSAF.
    Given a set of intervals, convert them into times.
    Parameters
    ----------
    inters: np.array(N-1, 2)
        A set of intervals.
    Returns
    -------
    times: np.array(N)
        A set of times.
    """
    return np.concatenate((inters.flatten()[::2], [inters[-1, -1]]), axis=0)


def downsample_frames(beat_frames, max_length=600):
    while len(beat_frames)>max_length:
        beat_frames = beat_frames[::2]
    return beat_frames


def remove_empty_segments(times, labels, th=2):
    """Removes empty segments if needed."""
    assert len(times) - 1 == len(labels)
    inters = times_to_intervals(times)
    new_inters = []
    new_labels = []
    j = 0
    for inter, label in zip(inters, labels):
        if inter[0] < inter[1] - th:
            new_inters.append(inter)
            new_labels.append(label)
        elif j == 0:
            if inter[0] != inter[1] :
                new_inters.append(inter)
                new_labels.append(label)
        j += 1
        
    return intervals_to_times(np.asarray(new_inters)), new_labels


class FileStruct:
    def __init__(self, audio_file):
        audio_file = Path(audio_file)
        self.track_name = audio_file.stem
        self.audio_file = audio_file
        self.ds_path = audio_file.parents[1]
        self.json_file = self.ds_path.joinpath('features', self.track_name
                                               + '.json')
        self.ref_file = self.ds_path.joinpath('references', self.track_name
                                              + '.jams')
        self.beat_file = self.ds_path.joinpath('features', self.track_name+'_beats_'
                                              + '.json')
        self.predictions_file = self.ds_path.joinpath('predictions', self.track_name
                                              + '.jams')
        self.audio_npy_file = self.ds_path.joinpath('audio_npy', self.track_name
                                              + '.npy')                                         

        
    def __repr__(self):
        """Prints the file structure."""
        return "FileStruct(\n\tds_path=%s,\n\taudio_file=%s,\n\test_file=%s," \
            "\n\json_file=%s,\n\tref_file=%s\n)" % (
                self.ds_path, self.audio_file, self.est_file,
                self.json_file, self.ref_file)

    def get_feat_filename(self, feat_id):
        return self.ds_path.joinpath('features', feat_id,
                                     self.track_name + '.npy')

def clean_tracklist_audio(data_path, annotations=None, tracklist_=[]):
    if tracklist_ == []:
        tracklist = librosa.util.find_files(os.path.join(data_path, 'audio'), ext=['wav', 'mp3', 'aiff', 'flac'])
    else:
        tracklist = tracklist_
    tracklist_clean = []
    for song in tqdm(tracklist):
        file_struct = FileStruct(song)
        if os.path.isfile(file_struct.beat_file) and os.path.isfile(file_struct.audio_npy_file):
            if annotations and os.path.isfile(file_struct.ref_file):
                tracklist_clean.append(song)
            elif not annotations:
                tracklist_clean.append(song)
    return tracklist_clean


def get_functional_labels_salami(file_struct, annot=-1):
    jam = jams.load(str(file_struct.ref_file), validate=False)
    try:
        annot = jam.search(namespace='segment_salami_function.*')[annot]
    except:
        print(file_struct.audio_file)
    ref_inters, ref_labels = annot.to_interval_values()
    ref_times =  intervals_to_times(ref_inters)
    ref_inters =  times_to_intervals(ref_times)
    (ref_inters, ref_labels) = mir_eval.util.adjust_intervals(ref_inters, ref_labels, t_min=0, t_max=ref_inters.max())
    duration = jam.file_metadata.duration
    return ref_labels, ref_times, duration


def read_references(audio_path, estimates, annotator_id=0, hier=False):
    """Reads the boundary times and the labels.

    Parameters
    ----------
    audio_path : str
        Path to the audio file

    Returns
    -------
    ref_times : list
        List of boundary times
    ref_labels : list
        List of labels

    Raises
    ------
    IOError: if `audio_path` doesn't exist.
    """
    # Dataset path
    ds_path = os.path.dirname(os.path.dirname(audio_path))


    if not estimates:
    # Read references
        try:
            jam_path = os.path.join(ds_path, 'references',
                                    os.path.basename(audio_path)[:-4] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, 'references',
                                    os.path.basename(audio_path)[:-5] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
    else:
        try:
            jam_path = os.path.join(ds_path, 'references/estimates/',
                                    os.path.basename(audio_path)[:-4] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)
        except:
            jam_path = os.path.join(ds_path, 'references/estimates/',
                                    os.path.basename(audio_path)[:-5] +
                                    '.jams')
            

            jam = jams.load(jam_path, validate=False)


    ##################
    low = True # Low parameter for SALAMI
    ##################


    if not hier:
        if low:  
            try:
                ann = jam.search(namespace='segment_salami_lower.*')[0]
            except:
                try:
                    ann = jam.search(namespace='segment_salami_upper.*')[0]
                except:
                    ann = jam.search(namespace='segment_.*')[annotator_id]
        else:
            try:
                ann = jam.search(namespace='segment_salami_upper.*')[0]
            except:
                ann = jam.search(namespace='segment_.*')[annotator_id]
        
        ref_inters, ref_labels = ann.to_interval_values()
        ref_times =  intervals_to_times(ref_inters)
        
        return ref_times, ref_labels

    else:

        list_ref_times, list_ref_labels = [], []
        upper = jam.search(namespace='segment_salami_upper.*')[0]
        ref_inters_upper, ref_labels_upper = upper.to_interval_values()
        
        list_ref_times.append( intervals_to_times(ref_inters_upper))
        list_ref_labels.append(ref_labels_upper)

        annotator = upper['annotation_metadata']['annotator']
        lowers = jam.search(namespace='segment_salami_lower.*')

        for lower in lowers:
            if lower['annotation_metadata']['annotator'] == annotator:
                ref_inters_lower, ref_labels_lower = lower.to_interval_values()
                list_ref_times.append( intervals_to_times(ref_inters_lower))
                list_ref_labels.append(ref_labels_lower)

        return list_ref_times, list_ref_labels





def read_references_2annot(audio_path, index):
    ds_path = os.path.dirname(os.path.dirname(audio_path))

    # Read references
    try:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-4] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)
    except:
        jam_path = os.path.join(ds_path, ds_config.references_dir,
                                os.path.basename(audio_path)[:-5] +
                                ds_config.references_ext)
        

        jam = jams.load(jam_path, validate=False)


    list_ref_times, list_ref_labels = [], []
    upper = jam.search(namespace='segment_salami_upper.*')[index]
    ref_inters_upper, ref_labels_upper = upper.to_interval_values()
    duration = jam.file_metadata.duration
    ref_inters_upper =  intervals_to_times(ref_inters_upper)
    #
    ref_inters_upper, ref_labels_upper =  remove_empty_segments(ref_inters_upper, ref_labels_upper)
    ref_inters_upper =  times_to_intervals(ref_inters_upper)
    (ref_inters_upper, ref_labels_upper) = mir_eval.util.adjust_intervals(ref_inters_upper, ref_labels_upper, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_upper)
    list_ref_labels.append(ref_labels_upper)


    lower = jam.search(namespace='segment_salami_lower.*')[index]
    ref_inters_lower, ref_labels_lower = lower.to_interval_values()
    ref_inters_lower =  intervals_to_times(ref_inters_lower)
    #(ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    ref_inters_lower, ref_labels_lower =  remove_empty_segments(ref_inters_lower, ref_labels_lower)
    ref_inters_lower =  times_to_intervals(ref_inters_lower)
    (ref_inters_lower, ref_labels_lower) = mir_eval.util.adjust_intervals(ref_inters_lower, ref_labels_lower, t_min=0, t_max=duration)
    list_ref_times.append(ref_inters_lower)
    list_ref_labels.append(ref_labels_lower)



    return list_ref_times, list_ref_labels, duration




def read_references_jsd(audio_path, level):

    file_struct = FileStruct(audio_path)
    
    jam = jams.load(str(file_struct.ref_file), validate=False)
   
    # Build hierarchy references
    for i in jam['annotations']['multi_segment']:
        bounds, labels, durations = [], [], []
        for bound in i['data']:
            if bound.value['level'] == level:
                bounds.append(bound.time)
                print(bound.value['label'])
                labels.append(bound.value['label'])
                durations.append(bound.duration)
    bounds.append(bounds[-1]+durations[-1])
    ref_int =  times_to_intervals(bounds)
    (ref_int, labels) = mir_eval.util.adjust_intervals(ref_int, labels, t_min=0, t_max=ref_int.max())
    bounds =  intervals_to_times(ref_int) 
    return bounds, labels


def get_duration(features_file):
    """Reads the duration of a given features file.

    Parameters
    ----------
    features_file: str
        Path to the JSON file containing the features.

    Returns
    -------
    dur: float
        Duration of the analyzed file.
    """
    with open(features_file) as f:
        feats = json.load(f)
    return float(feats["globals"]["duration"])


def create_json_metadata(audio_file, duration):
    if duration is None:
        duration = -1
    out_json = collections.OrderedDict({"metadata": {
        "versions": {"librosa": librosa.__version__,
                     "numpy": np.__version__},
        "timestamp": datetime.datetime.today().strftime(
            "%Y/%m/%d %H:%M:%S")}})
    # Global parameters
    out_json["globals"] = {
        "duration": duration,
        "sample_rate": 22050,
        "hop_length": 256,
        "audio_file": str(Path(audio_file).name)
        }
    return out_json


def write_beats(file_struct, beat_frames, duration):
    json_file = file_struct.beat_file
    if json_file.exists():
        print('BEAT FILE EXISTS')
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            #print(out_json['est_beats'])
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json = create_json_metadata(file_struct.audio_file, duration)
    out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    if json_file.exists():
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)


def read_beats(json_file):
    with open(json_file, 'r') as f:
        out_json = ujson.load(f)
    beat_strings = out_json["est_beats"].split('[')[1].split(']')[0].split(',')
    duration = float(out_json["globals"]["duration"])
    beat_times = [int(i) for i in beat_strings]
    return beat_times, duration

def read_beats_harmonix(file_struct):
    df = pd.read_csv(file_struct.annot_beat_file, sep="\t", header=None) 
    df = np.array(df) 
    beat_times = df[:,0]
    beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=256)
    return beat_frames

def write_features(features, file_struct, feat_id, feat_config, beat_frames, duration=None):
    # Save actual feature file in .npy format
    feat_file = file_struct.get_feat_filename(feat_id)
    json_file = file_struct.json_file
    feat_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, features)
    print('File saved')

    # Save feature configuration in JSON file
    if json_file.exists() and os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json =  create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json[feat_id] = {}
    variables = vars(getattr(feat_config, feat_id))
    for var in variables:
        out_json[feat_id][var] = str(variables[var])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    
    json_file = file_struct.beat_file
    if beat_frames != []:
        if json_file.exists():
            with open(json_file, 'r') as f:
                out_json = ujson.load(f)
        else:
            json_file.parent.mkdir(parents=True, exist_ok=True)
            out_json =  create_json_metadata(file_struct.audio_file, duration,
                                            feat_config)
        out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
        with open(json_file, "w") as f:
            ujson.dump(out_json, f, indent=4)


def update_beats(file_struct, feat_config, beat_frames, duration):
    json_file = file_struct.beat_file
    if json_file.exists():
        print('BEAT FILE EXISTS')
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            print(out_json['est_beats'])
    else:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        out_json =  create_json_metadata(file_struct.audio_file, duration,
                                        feat_config)
    out_json['est_beats'] = json.dumps([int(i) for i in beat_frames])
    with open(json_file, "w") as f:
        ujson.dump(out_json, f, indent=4)

    if json_file.exists():
        with open(json_file, 'r') as f:
            out_json = ujson.load(f)
            print(out_json['est_beats'])
    

def make_splits(data_path, val_data_path, p=.25, seed=42):
    tracklist = []
    for i in data_path:
        tracklist += clean_tracklist_audio(i, annotations=True)
    if val_data_path != None:
        valid_tracklist = clean_tracklist_audio(val_data_path, annotations=True)
        return tracklist, valid_tracklist
    if p == 0:
        return tracklist, tracklist
    else:
        train_tracklist, valid_tracklist = train_test_split(tracklist, test_size=p, random_state=seed)
        return train_tracklist, valid_tracklist
    

def verse_before_chorus(ref_labels):
    new_labels = []
    for i in range(len(ref_labels)-1):
        if ref_labels[i] == 'verse' and ref_labels[i+1]:
            new_labels.append('pre-chorus')
        else:
            new_labels.append(ref_labels[i])
    new_labels.append(ref_labels[-1])
    return new_labels


def chorus_rep(ref_labels):
    for i in range(len(ref_labels)-2):
        if ref_labels[i] == "chorus" and ref_labels[i+1] == "chorus" and ref_labels[i+2] == "chorus":
            return True
    return False


def check_durations(ref_inter, ref_labels):
    indices_verses = [i for i, x in enumerate(ref_labels) if x == "verse"]
    std_verse = np.std([ref_inter[i,1]-ref_inter[i,0] for i in indices_verses])

    indices_choruses = [i for i, x in enumerate(ref_labels) if x == "chorus"]
    std_chorus = np.std([ref_inter[i,1]-ref_inter[i,0] for i in indices_choruses])

    return max(std_verse, std_chorus) > 5
