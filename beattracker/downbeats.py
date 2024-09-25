import numpy as np

def get_downbeats(preds, pred_inds, odf_sal_full):
    """produces array of downbeat event times

    Args:
        preds np.array: array of prediction beats produced by agent system
        pred_inds list: list of indices that link predicted beats to positions in full odf
        odf_sal_full np.array: full array of odf scores

    Returns:
        np.array: subset of preds corresponding to downbeat times
    """
    odf_sal = odf_sal_full[pred_inds]
    
    max_mean = 0
    best_meter_hyp = 0
    test_ind = 0
    
    #ballroom dance is in either 3 or 4
    #iterate through first half of of the predictions to find the maximum of the mean odf value  / var 
    # For subsets of every three or four index
    for i in range(0, len(preds)//2):
        hyp_3 = np.mean(odf_sal[i::3])
        hyp_4 = np.mean(odf_sal[i::4])
        
        #assign meter hypothesis based on the subset with a higher value
        if hyp_4 >= hyp_3:
            meter_hyp = 4
        else:
            meter_hyp = 3
        
        #if metric is greater than existing max, update star_hyp and meter hypothesis
        if np.maximum(hyp_3,hyp_4) >max_mean:
            max_mean = np.maximum(hyp_3,hyp_4)/np.var(odf_sal[i::meter_hyp])
            best_meter_hyp = meter_hyp
            start_hyp = i
    
    #centered on the start hypothesis, take the downbeats by slicing based on meter hypothesis
    start_hyp = test_ind
    sec_1 = np.flip( preds[start_hyp::-best_meter_hyp])[:-1]
    sec_2 = preds[start_hyp::best_meter_hyp]
    
    downbeats = np.concatenate((sec_1,sec_2))

    return downbeats