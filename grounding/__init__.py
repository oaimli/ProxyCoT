# learning contextual grounding based on full articles

# direct-grounding (ProxyCoT-ZS):
# training the initial model on full contexts, with the reasoning traces from the teacher model based on proxy contexts

# reinforcement_0 (ablations):
# get the reasoning traces from the RL based model on proxy contexts
# use the reasoning traces to train the initial model on full contexts

# reinforcement_5 (ProxyCoT-RL):
# get the reasoning traces from the RL based model on proxy contexts
# use the reasoning traces to further train the RL-based model on full contexts