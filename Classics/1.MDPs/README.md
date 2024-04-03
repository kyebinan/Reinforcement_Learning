
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

