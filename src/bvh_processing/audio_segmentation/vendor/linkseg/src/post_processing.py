import numpy as np
import librosa
from scipy.stats import mode
import jams
import os

indices = {
    0: "silence",
    1: "verse",
    2: "chorus",
    3: "intro",
    4: "outro",
    5: "inst",
    6: "bridge",
}


indices_9classes = {
    0: "silence",
    1: "verse",
    2: "chorus",
    3: "intro",
    4: "outro",
    5: "inst",
    6: "bridge",
    7: "pre-chorus",
    8: "post-chorus"
}

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

def convert_labels(est_labels):
    if np.max(est_labels)>6:
        return [indices_9classes[i] for i in est_labels]
    else:
        return [indices[i] for i in est_labels]

def get_indices(beat_times, index, avg_future=6, avg_past=12):
    limit_left = 0
    limit_right = len(beat_times)-1
    for i in range(index, 0, -1):
        if beat_times[index]-beat_times[i]>avg_future:
            limit_left = i-1
            break
    for i in range(index, len(beat_times)):
        if beat_times[i]-beat_times[index]>avg_past:
            limit_right = i-1
            break
    return limit_left, limit_right



def pick_peaks_times(nc, beat_times, max_future=12, max_past=12, tau=0):
    
    peaks = []
    for i in range(1, nc.shape[0] - 1):
        limit_left_max, limit_right_max = get_indices(beat_times, i, max_future, max_past)
        try:
            max_left = np.max([nc[j] for j in range(i-1, limit_left_max, -1)])
        except:
            max_left = 0
        try:
            max_right = np.max([nc[j] for j in range(i+1, limit_right_max)])
        except:
            max_right = 10
            
        if max_left < nc[i] and nc[i] > max_right:
            if nc[i]>tau:
                peaks.append(i)
    return peaks



def post_process(audio_file, beat_times, duration, bound_curve, class_curves, max_past, max_future, tau):

    # We stack adjacent frames for boundary predictions, so we average adjacent beat times
    beat_times = [(beat_times[i] + beat_times[i+1])/2 for i in range(len(beat_times)-1)]

    assert len(beat_times) == len(bound_curve)
    
    bound_curve = bound_curve.reshape(-1)

    est_idxs = pick_peaks_times(bound_curve, beat_times, max_future=max_future, max_past=max_past, tau=tau)

    if len(est_idxs)>0:
        if est_idxs[0] != 0:

            est_idxs_ = [0] + list(est_idxs)
    else:
        est_idxs_ = [0] 
    est_idxs_ = est_idxs_ + [len(beat_times)-1]
    
    est_labels = []
    
    for i in range(len(est_idxs_)-1):
        bound_left = est_idxs_[i]
        bound_right = est_idxs_[i+1]
        class_predictions = np.argmax(class_curves[bound_left:bound_right], axis=-1)
        est_labels.append(mode(class_predictions)[0])

    est_idxs = [0] + [beat_times[int(i)] for i in est_idxs] + [duration]

    est_labels = convert_labels(est_labels)

    return est_idxs, est_labels
    


def export_to_jams(file_struct, duration, est_times, est_labels):
    ds_path = file_struct.ds_path
    pred_path = os.path.join(ds_path, 'predictions')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    jam = jams.JAMS()
    intervals = times_to_intervals(est_times)
    jam.file_metadata.duration = duration
    durations = [intervals[i,1]-intervals[i,0] for i in range(len(intervals))]
    ann = jams.Annotation(namespace='segment_open', time=0, duration=duration)
    for name, time, duration in zip(est_labels, est_times, durations):
        ann.append(value=name, time=time, duration=duration, confidence=None)
    jam.annotations.append(ann)
    jam.save(str(file_struct.predictions_file))
