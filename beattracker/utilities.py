import librosa
import os
import numpy as np
import mir_eval as mir

def load_file(file_path):
    """ Takes in file path 
    ln: input file path, optional annotations for training
    out: x (np. array representing wav)
        sr scalar: sample rate
    """
    x, sr = librosa.load(file_path, sr=None)
    return x, sr

def evaluate_results(predicted_beats, annotations): 
    """Produce evaluation metrics for input events

    Args:
        predicted_beats np.array: array of predicted beat times
        annotations np,array: two dimensional array of annotations, 
                                first column is beat times, second column is beat position

    Returns:
        dict: dictionary with keys f_measure and p_score
    """
    reference_beats = annotations[:,0]
    estimated_beats = predicted_beats
    evaluation_dict = mir.beat.evaluate(reference_beats, estimated_beats)
    eval_dict = {}
    f_m = evaluation_dict['F-measure']
    p_s = evaluation_dict['P-score']
     
    eval_dict['f_measure'] = f_m
    eval_dict['p_score'] = p_s
    return eval_dict
    