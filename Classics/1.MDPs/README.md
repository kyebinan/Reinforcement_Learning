# Markov Decesion Processes

MDPs are controlled Markov chains. We define below the state and action space, before actually
explaining what controlled Markov chains mean.

### State space.
The system is characterized at time t = 1, 2, . . . by its state st ∈ S. Unless otherwise
specified, S is finite and of cardinality S.

### Action space. 
In any state s ∈ S, the decision maker may select an action a ∈ As , that in
turn will impact the collected reward, and the system dynamics. Unless otherwise specficied, As
is finite, and we denote by A the cardinality of ∪s As .

### Controlled Markov chains. 
MDPs are controlled Markov chains. This means that the distribution of the state at time t + 1 depends on the past only through the state at time t and the
selected action. Let Ht = (s1 , a1 , s2 , a2 , . . . , at−1 , st ) denote the history up to time t. Then, we have: 
  - Formula

In the above, pt (·|s, a) represent the transition probabilities of the system at time t given that the state and the action at time t are (s, a).
The transition probabilities are called stationary if they do not depend on time t, i.e., pt (·|s, a) = p(·|s, a).

### Reward function. 
In most cases, we assume that the decision maker collects a deterministic reward at time t equal to rt (s, a) where (s, a) is the state and action pair at time t. 
Sometimes, it might be useful to consider random rewards, in which case we denote by qt (·|s, a) the reward distribution at time t given that the state and the action 
at time t are (s, a). The reward function is stationary if it does not depend on t.

## Three classes of MDPs

  - Finite-time horizon MDPs.
  - Stationary MDPs with terminal state.
  - Infinite-time horizon discounted MDPs

