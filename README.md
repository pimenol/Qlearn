# BlockWorld Q-learning Assistant

## Description:
This project aims to provide a solution to the classic BlockWorld problem using Q-learning, an approach from reinforcement learning. The BlockWorld problem involves moving crates between stacks while facing the challenge of occasional memory lapses, simulating real-world scenarios where errors can occur.

## Features:

### Q-learning Algorithm Implementation: 
The core of this project is the implementation of the Q-learning algorithm. The algorithm learns from past experiences to make better decisions over time, helping to optimize crate movements in the BlockWorld environment.
### BlockWorld Environment: 
The BlockWorld environment, represented by the BlockWorldEnv class, provides the setting for crate manipulation. It offers methods for resetting the environment and executing actions, crucial for training and testing the Q-learning agent.
### Training and Evaluation: 
The train() method is responsible for training the Q-learning agent using the BlockWorld environment. The performance of the trained agent is evaluated based on its ability to solve a variety of BlockWorld instances efficiently, considering both the number of steps taken and the time elapsed.
