
# Markov Decision Processes

Markov Decision Processes (MDPs) provide a formal framework for making decisions in environments where outcomes are partially random and partially under the control of a decision-maker. This chapter delves into the core aspects of MDPs, including their definition, various classes, and solution methods.

## 2 Markov Decision Processes

MDPs are essentially controlled Markov chains. Here, we outline the state and action space before diving into what it means to control Markov chains.

### State Space

The system's state at time `t` is represented by `st` in set `S`. Unless specified otherwise, `S` is finite with cardinality `S`.

### Action Space

For any state `s ∈ S`, the decision-maker can choose an action `a ∈ As`, impacting the collected reward and the system's dynamics. Unless specified otherwise, `As` is finite, and `A` is the cardinality of the union of all `As`.

### Controlled Markov Chains

MDPs are controlled Markov chains, implying that the future state's distribution only depends on the current state and action, not the full history up to that point. This is described by the transition probabilities `pt(s|st, a)`.

### Reward Function

A deterministic reward `rt(s, a)` is assumed to be collected at time `t`, which can be deterministic or stochastic, represented by `qt(·|s, a)` for the reward distribution.

## 2.1 Three Classes of MDPs

### Finite-time Horizon MDPs

The objective here is to maximize the expected reward for the first `T` steps.

### Stationary MDPs with Terminal State

These problems assume stationary transition probabilities and rewards, along with a terminal state after reaching which no rewards are collected.

### Infinite-time Horizon Discounted MDPs

Here, rewards are discounted by a factor `λ ∈ (0, 1)`, and the goal is to maximize the expected discounted reward over an infinite horizon.

## 2.2 Solving Finite-time Horizon MDPs

This involves procedures to evaluate the reward of a given policy and to identify an optimal policy along with its reward through policy evaluation and the optimal policy definition.

### Policy Evaluation

For a given Markovian deterministic policy `π`, we can compute the state value functions and subsequently the expected reward under `π`.

### The Optimal Policy

The value function of an MDP is defined as the state value function of the optimal policy. This can be computed using a backward induction process known as Dynamic Programming (DP).

## 2.3 Solving Stationary MDPs with Terminal State and Infinite-time Horizon Discounted MDPs

Both types of problems can be approached using similar methods, with the main difference being the consideration of a discount factor in infinite-time horizon MDPs.

### Policy Evaluation

For a stationary Markovian deterministic policy `π`, the state value function `Vπ` maps the current state to the expected discounted reward collected under `π`.

### The Optimal Policy

The value function `V` is the state value function of an optimal policy, found as the unique solution of Bellman’s equations.

## Solution Methods

For finite-time horizon MDPs, we can use policy evaluation to assess the reward of a given policy and dynamic programming to find an optimal policy. Stationary and infinite-time horizon discounted MDPs can be addressed with policy iteration and value iteration algorithms, aiming to find policies that maximize expected rewards under the given conditions.


## The Robot Vacuum Cleaner

You work for a firm designing automated vacuum cleaners and have been tasked to beta-test your newest cleaner over the period of 90 days at your home. You live in a square flat that is 36 square meters and will be starting the cleaner each day. The flat is divided into 36 square cells each of surface 1 m2. Your flat is characterized by the fact that provided it was cleaned the previous day, each morning there is new dust distributed over your entire apartment drawn independently (over days) according to some distribution q(·). Each square cell of your flat is either clean, dusty, or very dusty each morning. Hence, the distribution q(·) describes the joint probability that each cell of your flat is clean, dusty, or very dusty each morning.

Each morning, before the robot starts moving, it observes the state of each cell (clean, dusty, or very dusty). Then, the robot decides on a path through your apartment. The choice of the path is constrained by the facts that: (i) the robot always starts in the same position, the bottom left cell of your flat; (ii) it visits every unclean cell; and (iii) it returns to the bottom left at the end of the path. The robot can move vertically and horizontally but cannot move diagonally. The robot spends 1 minute on an already clean cell of your flat, 2 minutes on a dusty cell, and 3 minutes on a very dusty cell. We assume that the time it takes for the robot to move from one cell to neighboring cells is negligible. If a cell has already been visited, assume that the robot still spends one minute there. Note that the path taken by the robot is decided upfront each morning and cannot be revised while the robot is cleaning. The objective is to design a robot minimizing the expected time spent to clean the flat over the 90 days.

## (a) Model the problem as an MDP. Do not solve the MDP. 

(Remember that the robot has access to the amount and location of the dust in the flat each morning). 

## Your employer does not like the assumption that the robot knows the amount and location of the dust. 

To keep your job as the firm’s leading robot vacuum designer, you make another attempt.

## (b) Model the problem as an MDP. Do not solve the MDP. 

This time assume the robot does not have access to the amount and location of the dust. Here again, the path has to be decided upfront before the robot starts moving. [Hint: use random rewards]

## Solution

### (a) The state space may be defined as follows. 

Let V = {s = (s1, s2) ∈ Z^2 | 0 ≤ si ≤ 5} and N = {1, 2, 3} defines how dusty the part of V in the morning. Recall that the notation NV denotes functions from V to N. Then S = NV, and each element of s ∈ S is a function s = n(·) taking vertices as inputs and dustiness ∈ {1, 2, 3} as output. We define the graph G by letting V be its vertices and drawing an edge between every adjacent vertex (we define adjacency by the coordinates of S differing by at most one entry by magnitude exactly 1). The actions can now be written as As = {Paths over the graph G that begin at (0, 0) and terminate at (0, 0) and visit every unclean point of G} where a point v is unclean if n(v) = 1. If the action a is a path, the rewards are r(s, a) = −n(v) + max{|occurrences of v ∈ a| − 1, 0} for v∈a, which is the negative total time spent cleaning the apartment. To be precise, summation above treats a as a set and counts multiple occurrences of the same vertex once. Repeated visits are accounted for by the term containing the max. The goal is to maximize Eπ[Σ r(st, at)] from t=0 to 89. Finally, the transitions are given by p(s'|s, a) = q(s').

### (b) There is no longer a state space/singleton state – the problem is a bandit. 

Otherwise, the problem can be modeled exactly as above, with the key distinction that S no longer holds significance as a state space. Instead, the dustiness level of your flat s becomes an auxiliary random variable only used to compute the rewards (which are now random functions of your actions).

