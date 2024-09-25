import numpy as np


class Agent:
    """_summary
    stores the state of agents in the agent system
    simple class with getter and setter methods
    In: temp_hyp: scalar
        prediction: scalar
        history: list
        score: scalar
        beats_to_pred: scalar (tracks steps when interpolating)
        odf_ind: list   (used in downbeat tracker. Logs the full odf index of each event added to agent history)
    """
    def __init__(self, temp_hyp, prediction, history, score, beats_to_pred, odf_ind):
        self.temp_hyp = temp_hyp
        self.prediction = prediction
        self.history = history
        self.score = score
        self.beats_to_pred = beats_to_pred
        self.odf_ind = odf_ind
        self.delete = False
    
    def mark_delete(self):
        self.delete = True
        
    def increment_prediction(self):
        self.prediction += self.temp_hyp
        self.beats_to_pred += 1
    
    def replace_prediction(self, new_pred):
        self.prediction = new_pred
        self.beats_to_pred = 1

    def update_temp_hyp(self, adj_error):
        self.temp_hyp += adj_error
    
    def update_score(self, adj_sal):
        self.score += adj_sal
    
    def copy_agent(self):
        copy = Agent(self.temp_hyp, self.prediction, 
                     self.history.copy(), self.score, 
                     self.beats_to_pred, self.odf_ind.copy())
        return copy
    
    def get_prediction(self):
        return self.prediction
    
    def get_temp_hyp(self):
        return self.temp_hyp
    
    def get_last_event(self):
        return self.history[-1]

    def get_beats_to_pred(self):
        return self.beats_to_pred
    
    def get_temp_hyp(self):
        return self.temp_hyp
    
    def get_history(self):
        return self.history

    def get_odf_ind(self):
        return self.odf_ind
    
    def get_delete(self):
        return self.delete
    
    def get_score(self):
        return self.score
    
    

class AgentList:
    """Class representing list of agents
    In: clusters: np.array of tempo hypotheses
        events: np.array of onset events
        event_odf_ind: list of indices that link each event to position in full event
        startup_sec: scalar
    """
    def __init__(self,clusters, events, event_odf_ind, startup_sec):
        self.clusters = clusters
        self.events = events
        self.startup_sec = startup_sec
        self.agents = []
        
        #array of indices in the original odf for each onset event
        #used to link peaks in odf to corresponding lfc for scaling
        self.event_odf_ind = event_odf_ind
    
    def intialize_agents(self):
        startup_ind = 0
        #find first index outside of startup period
        while self.events[startup_ind] <= self.startup_sec and startup_ind+1 < len(self.events):
            startup_ind += 1
        
        #intialize one agent for each tempo hypothesis and event in the startup period
        for temp_hyp in self.clusters:
            for i, event_time in enumerate(self.events[0:startup_ind]):
                odf_ind = self.event_odf_ind[i]
                prediction = event_time + temp_hyp
                history = [event_time]
                score = 0
                beats_to_pred = 1
                odf_ind = [odf_ind]
                
                new_agent = Agent(temp_hyp, prediction, history, score, beats_to_pred, odf_ind)
                self.agents.append(new_agent)
    
    def add_agents(self, new_agents):
        self.agents = self.agents + new_agents
    
    def remove_deleted(self):
        agents_keep = []
        for agent in self.agents:
            if not agent.get_delete():
                agents_keep.append(agent)
        
        self.agents = agents_keep
    
    def get_agents(self):
        return self.agents
    
    def get_best_agent(self):
        best_agent = sorted(self.agents, key=lambda x: x.get_score(), reverse=True)[0]
        return best_agent


def agent_track(clusters, events, odf_score, frame_sec, event_odf_ind, agent_param_dict):
    """Runs agent systme

    Args:
        clusters np.array: tempo hypotheses
        events np.array: onset events
        odf_score np.array: ODF value associated with each peak. has the same dimensionality as events
        frame_sec scalar: the length of a frame in seconds (used to find odf index of interpolated beats)
        event_odf_ind np.array: list of indices that link each event to position in full event
        agent_param_dict dict: dictionary containing the parameters needed to run agent system

    Returns:
        np.array: retrunrs
        list: list of indices that link each event to position in full odf
    """
    startup_sec = agent_param_dict['su']
    inner_win = agent_param_dict['iw']
    cf = agent_param_dict['cf']
    time_out = agent_param_dict['to']
    
    agents = AgentList(clusters, events ,event_odf_ind, startup_sec)
    agents.intialize_agents()

    #iterate through each event and each agent
    for i, event in enumerate(events):
        new_agents = []
        
        for agent in agents.get_agents():
            #calc outer windows based on beat interval
            #from Dixon, 2006, assymettry reflects expressive reductions in tempo are more common and larger than tempo increases
            temp_hyp = agent.get_temp_hyp()
            outer_win_pre = .2*temp_hyp
            outer_win_post = .4*temp_hyp
            
            #If the gap between an event and the last beat an agent is tracked is greater than some threshold, delete it
            last_event = agent.get_last_event()
            if event - last_event > time_out:
                agent.mark_delete()
            
            else:
                #incrementally add tempo hypothesis until the event occurs before the upper bound of outer window
                #while incrementing the var tracking the number of beats to the prediction
                pred = agent.get_prediction()
                while pred + outer_win_post <= event:
                    agent.increment_prediction()
                    pred = agent.get_prediction()
                
                
                #if the event falls inside the outer window, then
                if pred - outer_win_pre <= event <= pred + outer_win_post:
                    #if prediction does not fall in inner window, new agent is created with no updates
                    if np.abs(pred - event) > inner_win:
                        new_agent = agent.copy_agent()
                        new_agents.append(new_agent)
                    
                    #add interpolated beats to history using beat interval
                    #not in dixon paper, assume that if the agents are interpolating beats, we want to 
                    #add them to history as well
                    interp_beats = agent.get_beats_to_pred()
                    last_event = agent.get_last_event()
                    history = agent.get_history()
                    odf_ind_hist = agent.get_odf_ind()
                    
                    for j in range(interp_beats - 1):
                        interp_beat = (j + 1) * temp_hyp + last_event
                        
                        odf_ind = int(round(interp_beat/frame_sec))
                        history.append(interp_beat)  
                        odf_ind_hist.append(odf_ind)      
                    
                    #update beat interval based on error, store prediction for next beat, and add the event to history
                    error = event - pred
                    agent.update_temp_hyp(error/cf)
                    upd_pred = event + agent.get_temp_hyp()
                    agent.replace_prediction(upd_pred)
                    
                    history.append(event)
                    
                    #if agent overshoots event, relative error is based on upper bound of outerwindow
                    #otherwise, relative error based on lower bound of outer window
                    if error < 0:
                        relative_error = np.abs(error)/outer_win_post
                    else:
                        relative_error = error/outer_win_pre
                    
                    #Salience value based on odf val
                    adj_sal = (1-relative_error/2)*odf_score[i]
                    agent.update_score(adj_sal)
                    
                    #convert event index to index in full odf and store
                    odf_ind = event_odf_ind[i]
                    odf_ind_hist.append(odf_ind)
                    
        #add new agents created from preds in outer window
        agents.add_agents(new_agents)
        
        #mark similiar agents for deletion
        #if pair of agents has within 10ms of tempo var and 20ms of predicted beat dif, pick agent with highest score
        for k, agent_k in enumerate(agents.get_agents()):
            if not agent_k.get_delete():
                for l, agent_l in enumerate(agents.get_agents()):
                    if (k != l and np.abs(agent_k.get_temp_hyp() - agent_l.get_temp_hyp()) < .010  
                        and np.abs(agent_k.get_prediction() - agent_l.get_prediction()) < .020):
                        agent_l.mark_delete()
        
        agents.remove_deleted()
        
    #return highest scoring agent. First case protects against case where all agents timeout
    if len(agents.get_agents()) == 0:
        prediction = {}
        prediction_ind = []
        
    else:
        prediction = np.array(agents.get_best_agent().get_history())
        prediction_ind = agents.get_best_agent().get_odf_ind()
    
    return prediction, prediction_ind


