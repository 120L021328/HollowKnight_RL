# Playing Hollow Knight with Reinforcement Learning

The project uses Deep Q-Network to learn how to play boss fight in Hollow Knight.

I am currently primarily trying to train the agent with Hornet boss fight, 
but it should be able to run on most Hall of Gods Boss with only one boss health bar.

You need to install the Satchel and EnemyHPBar Mod to correctly run (so it can recognize boss HP), 
I have made some modifications (custom HP bar color) 
for more stable recognition, the mod files can be found in Managed folder.

_________________________

## Usage
run Hollow knight in window mode, make sure use the following keymap settings:
![Keyboard settings](resources/keymaps.png)
Currently, only up, down, left, right, jump, attack, and dash are used, so others doesn't matter at this time.

Then, manually walk to one of the Hall of Gods statues, until you see the 'Challenge' prompt, 
run the following from terminal with virtual environment and all required packages installed:

```
python train.py
```

It will take over keyboards to control the game, if you want to force quit in the middle, Ctrl+C in terminal will work

___________________________

## Project Tentative Plan:
- [ ] Update Project Structure
  - [x] Add requirements.txt, mod files
  - [x] Add instructions to train
  - [ ] Add more docstrings and typehint in code
- [ ] Train an agent that can defeat Hornet
- [ ] add code for additional functionalities (like saving buffer, model...)
- [ ] improve the algorithm


Project inspired by https://github.com/ailec0623/DQN_HollowKnight <br>
that is a very interesting project, and the author has already defeated Hornet with a trained agent. However, that project uses CE and windows API heavily, which I am less familiar with, so I decided to make one on my own.
