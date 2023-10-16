# Reinforcement-Learning-Surround
This project is for the master´s course in Reinforcement Learning, ministry by Flávio Codeço Coelho at FGV-EMAp

# Summary

- [The Problem](#the-problem)
- [The Environment](#the-environment)
    - [Observation Space](#observation-space)
    - [Action Space](#action-space)
    - [Game Rules](#game-rules)
- [Solving the Problem](#solving-the-problem)
    - [1 Algorithm](#1-algorithm)
        - [Training](#training)
        - [Testing](#testing)
        - [Results](#results)
    - [2 Algorithm](#2-algorithm)
        - [Training](#training-1)
        - [Testing](#testing-1)
        - [Results](#results-1)
- [Conclusion](#conclusion)
- [Usage Guide](#usage-guide)
- [References](#references)

# The Problem

The goal is to solve the Atari game Surround (shown in the figure below) using two different algorithms. The goal is to surround the opponent snake without hitting the walls or the opponent snake. The game ends when one of the snakes hits the wall or the opponent snake.


<div align="center">
	<img src = "report/surround_atari.png" width=40%> 
</div>

# The Environment

The enviroment used in the original Surround game is a 11 x 11 grid where two snakes move around, leaving a trail behind them. On the Atari 2600 version, just one player controls the snake and the other snake is controlled by the computer. 

The version used in this project is a 16 x 16 grid where the two snakes are controlled by the users inputs (describe in <a href="#usage-guide">Usage Guide</a>).

## Observation Space

The observation space is a 16 x 16 x 3 matrix where each layer is filled with 0 for nothing and 1 for objects. The first layer is filled with 1 for the walls, the second layer is filled with 1 for the player 1 snake and the third layer is filled with 1 for the player 2 snake.

## Action Space
Each action is represented by a number from 0 to 4, as shown in the table below.

<div align = "center">
<table>
  <tr>
    <th>Value</th> <th>Meaning</th>
  </tr>
  <tr>
    <td>0</td> <td>stay</td>
  </tr>
  <tr>
    <td>1</td> <td>right</td>
  </tr>
  <tr>
    <td>2</td> <td>up</td>
  </tr>
  <tr>
    <td>3</td> <td>left</td>
  </tr>
  <tr>
    <td>4</td> <td>down</td>
  </tr>
</table>
</div>

## Game Rules
The snakes movement are based on the player perspective, so if the snake is moving to the right, the action right will be considered as the stay action.

A snake can't move to the opposite direction of its current movement. For example: if the snake is moving to the right, it can't move to the left, so the action left will be considered as the stay action.

# Solving the Problem

## 1 Algorithm

### Training

### Testing

### Results

## 2 Algorithm

### Training

### Testing

### Results

# Conclusion

# Usage Guide

# References
