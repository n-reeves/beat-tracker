import numpy as np


class Cluster:
    """Simple class storing the state of each cluster
    In: cluster_id scalar: unique id for cluster
        ioi: first ioi scalar added to the cluster. Subsequent iois are added with add_cluster
    """
    def __init__(self, cluster_id, ioi):
        self.cluster_id = cluster_id
        self.iois = [ioi]
        self.cluster_center = ioi
        self.delete = False
        self.score = 0
    
    def update_center(self):
        self.cluster_center = np.mean(self.iois)
    
    def add_ioi(self, ioi):
        self.iois.append(ioi)
        #whenever ioi is added, update the center
        self.update_center()
    
    def add_cluster(self, new_iois):
        self.iois = self.iois + new_iois
        #whenever clusters are joined, update center
        self.update_center()
    
    def mark_delete(self):
        self.delete = True
    
    def update_score(self, val):
        self.score += val
        
    def get_iois(self):
        return self.iois
    
    def get_delete(self):
        return self.delete
    
    def get_center(self):
        return self.cluster_center
    
    def get_score(self):
        return self.score
    


class ClusterList:
    """Class storing state of the list of clusters
    In: bpm_lower scalar: lower bound of bpm range
        bpm_upper scalar: upper bound of bpm range
        top_n scalar: number of clusters to return
    """
    def __init__(self, bpm_lower, bpm_upper, top_n):
        self.cluster_counter = 0
        self.cluster_list = {}
        #convert bpm to ioi time
        self.ioi_lower = 60000/bpm_upper * .001
        self.ioi_upper = 60000/bpm_lower * .001
        self.top_n = top_n
        
    def add_cluster(self, ioi):
        #intialize cluster
        new_cluster = Cluster(self.cluster_counter,ioi)
        self.cluster_list[self.cluster_counter] = new_cluster
        self.cluster_counter += 1
    
    def remove_deleted(self):
        #remove delted clusters
        new_cluster_list = {}
        for cluster_id in self.cluster_list:
            cluster = self.get_cluster(cluster_id)
            if not cluster.get_delete():
                new_cluster_list[cluster_id] = cluster
        
        self.cluster_list = new_cluster_list
    
    def bpm_filt(self):
        #apply BPM filter
        for cluster_id in self.cluster_list:
            cluster = self.get_cluster(cluster_id)
            if cluster.get_center() < self.ioi_lower or cluster.get_center() > self.ioi_upper:
                cluster.mark_delete()
        self.remove_deleted()
    
    def return_top_n(self):
        sorted_clusters = sorted(self.cluster_list.items(), key=lambda c: c[1].get_score(), reverse=True)
        return sorted_clusters[:self.top_n]
    
    def get_ids(self):
        return self.cluster_list.keys()
    
    def get_cluster(self, cluster_id):
        return self.cluster_list[cluster_id]
    


def cluster_ioi(ioi_ar, cluster_param_dict, top_n=5):
    """_summary_

    Args:
        ioi_ar np.array: 1d array of inter-onset intervals between events found with onset detector
        cluster_param_dict dict: dictionary of parameters used in clustering
        top_n (int, optional): Number of tempo hypotheses returned. Defaults to 5.

    Returns:
        list: list of top_n cluster centers
    """
    cluster_width = cluster_param_dict['cw'] 
    bpm_lower = cluster_param_dict['bpm_l']
    bpm_upper = cluster_param_dict['bpm_u']
    
    #initialize cluster list
    clusters = ClusterList(bpm_lower, bpm_upper, top_n)
    #Assign points to clusters. If a point is within the clusterwidth of an exisiting cluster, add it, 
    #otherwise create a new cluster and add it to that
    for ioi in ioi_ar:
        to_add = True
        for cluster_id in clusters.get_ids():
            cluster = clusters.get_cluster(cluster_id)
            if to_add and np.abs(cluster.get_center() - ioi) < cluster_width:
                cluster.add_ioi(ioi)
                to_add = False
        if to_add:
            clusters.add_cluster(ioi)

    #iterate through clusters and combine clusters that have absolute difference in mean time below the cluster width 
    for cluster_id_i in clusters.get_ids():
        cluster_i = clusters.get_cluster(cluster_id_i)
        if not cluster_i.get_delete():
            for cluster_id_j in clusters.get_ids():
                cluster_j = clusters.get_cluster(cluster_id_j)
                dist = np.abs(cluster_i.get_center() - cluster_j.get_center())
                #don't add deleted clusters or the same clusters, or clusters with centers > cluster width 
                if cluster_id_i != cluster_id_j and not cluster_j.get_delete() and dist < cluster_width: 
                    cluster_i.add_cluster(cluster_j.get_iois())
                    cluster_j.mark_delete()


    #remove deleted clusters
    clusters.remove_deleted()
    
    #compute relationship scores for each group
    #if cluster center i is an integer multiple of cluster center j by a factor of n (absolute dif below cluster width)
    #the score for cluster i increases by a relationship factor dependent on the integer multiple * the number of iois in cluster j
    for cluster_id_i in clusters.get_ids():
        cluster_i = clusters.get_cluster(cluster_id_i)
        for cluster_id_j in clusters.get_ids():
            cluster_j = clusters.get_cluster(cluster_id_j)
            for n in range(1,9):
                #Add to cluster i score based if it is an integer multiple (1-8) of cluster j
                if np.abs( cluster_i.get_center()  - n*cluster_j.get_center()) < cluster_width:
                    if n >= 1 and n <= 4:
                        rel_factor = 6-n
                    elif n >= 5 and n <= 8:
                        rel_factor = 1
                    else:
                        rel_factor = 0
                    cluster_i.update_score(rel_factor * len(cluster_j.get_iois()))
    
    #remove clusters with bpms outside of range
    clusters.bpm_filt()
    
    #return top n clusters
    top_n_clusters = clusters.return_top_n()
    
    #return the centers of the top n clusters as tempo hypotheses
    tempo_hyp = np.array([tup[1].get_center() for tup in top_n_clusters] )

    return tempo_hyp