# Hupo

### Usage
```julia
julia> include("stateValue.jl")

julia> computeStateValue!(n_epochs = 10000, train_top = true)

julia> play(mcts_net_top)
```

#### How to play
- use arrow keys to move
- use number keys to pass
- use `play(opponent, true)` to play as top player
- use `play(opponent, delay = 0.5)` to set the delay to 0.5 seconds
  - use `delay = NaN` to confirm each opponent's action by pressing Enter
- *works on Linux only, support for Windows coming soon ;-)*

### Rules
https://intersob.math.muni.cz/oldfoto/2/zadani/ukol07.pdf  
but slightly different

### Worth reading
http://karpathy.github.io/2016/05/31/rl/  
https://medium.freecodecamp.org/deep-reinforcement-learning-where-to-start-291fb0058c01  
https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.109.810&rep=rep1&type=pdf   
http://ruder.io/optimizing-gradient-descent/

### Aim
1. Create a bot that is able to beat a human controlling all three stones at once.
2. Create three bots controlling one stone each. Use "team spirit" parameter.

### Steps
1. Create a bot that is beating human in simple hupo game with only one middle stone for each player. First player should win 100 percent of time.

### TODO
1. optimize speed
2. save "generations" of nets
3. ~~add environment for human vs net~~
<!-- 4. add probability information to game printing -->
5. try ADAM
6. try different architectures
7. try randomizing initial state
8. mini-batching
