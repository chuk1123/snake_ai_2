# 🐍 Snake AI - Deep Q-Learning Agent

> An AI agent trained with Deep Q-Learning to master the classic Snake game using PyTorch and Pygame

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-red.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.1.2-green.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎮 About

This project implements a Deep Q-Learning (DQN) reinforcement learning agent that learns to play Snake through trial and error. The agent uses a neural network to predict optimal moves and improves over time using experience replay and epsilon-greedy exploration.

### Key Features

- **🧠 Deep Q-Learning**: Neural network-based Q-learning with experience replay
- **📊 Live Training Visualization**: Real-time plotting of scores and performance
- **🎯 Optimized State Representation**: 26-feature input including multi-distance danger detection
- **💾 Model Persistence**: Save and load trained models for inference
- **🕹️ Human Playable Version**: Play the game yourself to compare with AI performance

## 🤖 How It Works

### Reinforcement Learning Approach

The agent uses **Deep Q-Learning (DQN)**, where:
- **Q-values** represent expected future rewards for each action
- A **neural network** approximates the Q-function
- **Experience replay** stores and samples past experiences for stable learning
- **Epsilon-greedy** exploration balances trying new moves vs. exploiting learned behavior

### State Representation (26 Features)

**1. Danger Detection (15 features)**
- Detects obstacles at 5 different distances: 1, 2, 3, 4, and 5 blocks ahead
- Three directions per distance: straight, left turn, right turn
- Binary encoding: 1 = danger, 0 = safe

**2. Current Direction (4 features)**
- One-hot encoding: [left, right, up, down]

**3. Food Location (4 features)**
- Relative position: [food_left, food_right, food_up, food_down]
- Binary: 1 if food is in that direction

**4. Current Direction Vector (3 features)**
- Direction values: left, right, up, down

### Neural Network Architecture

```
Linear_QNet:
  Input Layer:  26 neurons (state features)
  Hidden Layer: 256 neurons (ReLU activation)
  Output Layer: 3 neurons (Q-values for: straight, left turn, right turn)
```

**Optimizer**: Adam
**Learning Rate**: 0.001
**Loss Function**: Mean Squared Error (MSE)

### Training Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Epsilon Decay** | 80 games | Exploration → exploitation transition |
| **Gamma (Discount)** | 0.9 | Future reward weight |
| **Memory Size** | 100,000 | Experience replay buffer |
| **Batch Size** | 1,000 | Training sample size |
| **Learning Rate** | 0.001 | Neural network update rate |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/chuk1123/snake_ai_2.git
   cd snake_ai_2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Train a New Agent

Run the training script to train a new AI agent from scratch:

```bash
python agent.py
```

**Training Process**:
- Agent plays games repeatedly, learning from each experience
- Live plot shows score progression over games
- Models automatically saved to `model/` directory
- Training can be stopped anytime (Ctrl+C) and resumed later

**Expected Training Time**: Significant improvement after 100-200 games, optimal performance around 500-1000 games.

#### Test a Trained Model

Load and watch a pre-trained model play:

```bash
python testing.py
```

This loads the saved model and runs inference (no learning).

#### Play as Human

Want to compare your skills with the AI?

```bash
python snake_game_human.py
```

**Controls**:
- Arrow Keys: Move the snake
- Goal: Eat food, avoid walls and yourself

## 📊 Performance

### Training Results

- **Record Score**: Varies by training run (typically 40-80+ on 800x800 grid)
- **Average Score (late training)**: 20-40
- **Learning Curve**: Exponential improvement in first 100-200 games, then gradual optimization

### Model Files

| File | Description |
|------|-------------|
| `model/best2.pth` | Best performing model checkpoint (~61KB) |
| `model/model2.pth` | Alternative checkpoint (~31KB) |

## 📁 Project Structure

```
snake_ai_2/
├── agent.py               # Main training script with RL agent
├── model.py               # Neural network (Linear_QNet) and Q-trainer
├── game.py                # Pygame Snake game environment
├── helper.py              # Matplotlib visualization for training
├── snake_game_human.py    # Human-playable Snake version
├── testing.py             # Inference script for trained models
├── model/                 # Saved trained models
│   ├── best2.pth
│   └── model2.pth
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── LICENSE                # MIT License
```

## 🧪 Technical Details

### Reward Function

The agent receives:
- **+10 points**: For eating food
- **-10 points**: For colliding (game over)
- **0 points**: For valid moves (survival)

### Training Algorithm

1. **Get current state** (26 features)
2. **Choose action** (epsilon-greedy):
   - Random (exploration) with probability ε
   - Best Q-value (exploitation) with probability 1-ε
3. **Execute action** in environment
4. **Observe reward** and next state
5. **Store experience** in replay memory
6. **Sample batch** from memory and train neural network
7. **Repeat** until convergence

### Multi-Distance Danger Detection

Unlike simple 1-block lookahead, this agent detects dangers at **5 different distances** (1-5 blocks ahead), allowing:
- Better path planning
- Avoiding traps earlier
- Smarter food-seeking behavior

## 🔧 Customization

### Modify Training Parameters

Edit values in `agent.py`:

```python
# Number of games to train
MAX_GAMES = 500

# Exploration rate decay
epsilon = 80 - agent.n_games
```

### Adjust Network Architecture

Modify `model.py`:

```python
# Change hidden layer size
self.linear1 = nn.Linear(input_size, 256)  # Try 128, 512, etc.
```

### Change Game Settings

Edit `game.py`:

```python
# Grid size
BLOCK_SIZE = 20

# Game speed (for human play)
SPEED = 20  # Lower = slower, higher = faster
```

## 📈 Training Tips

1. **Let it run**: Agent needs 100+ games to show meaningful learning
2. **Watch the plot**: Score should trend upward over time
3. **Epsilon decay**: Early games are random (exploration), later games use learned policy
4. **Save checkpoints**: Models are auto-saved, so you can resume training anytime
5. **Compare models**: Try different hyperparameters and compare performance

## 🎯 Future Improvements

Potential enhancements:
- **Double DQN** or **Dueling DQN** for more stable learning
- **Prioritized Experience Replay** to focus on important experiences
- **Convolutional layers** for visual input (raw pixels)
- **Larger grid** for more challenging gameplay
- **Curriculum learning** (start small, gradually increase difficulty)
- **Multi-agent training** (competitive Snake)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Hyperparameter tuning
- Alternative RL algorithms (PPO, A3C)
- Better state representations
- Performance benchmarking

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Created by [Kevin Chu](https://github.com/chuk1123)
- Built with [PyTorch](https://pytorch.org/) and [Pygame](https://www.pygame.org/)
- Inspired by Deep Q-Learning papers and reinforcement learning research

## 📚 Learning Resources

Want to learn more about Deep Q-Learning?

- **[Playing Atari with Deep RL (DeepMind)](https://arxiv.org/abs/1312.5602)** - Original DQN paper
- **[Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html)** - The RL bible
- **[OpenAI Spinning Up](https://spinningup.openai.com/)** - Modern RL guide
- **[PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)** - Official PyTorch DQN tutorial

---

**Made with 🐍🧠 by chuk1123**

*Watch the AI learn, evolve, and master Snake!*
