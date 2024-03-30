from collections import namedtuple


"""namedtuple is used to create a special type of tuple object. Namedtuples
always have a specific name (like a class) and specific fields.
In this case I will create a namedtuple 'Experience',
with fields: state, action, reward,  next_state, done.
Usage: for some given variables s, a, r, s, d you can write for example
exp = Experience(s, a, r, s, d). Then you can access the reward
field by  typing exp.reward"""
Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])
