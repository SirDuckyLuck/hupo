# Hupo
### Rules
https://intersob.math.muni.cz/oldfoto/2/zadani/ukol07.pdf
but slightly different

### Inspiration
http://karpathy.github.io/2016/05/31/rl/

### Aim
1. Create a bot that is able to beat a human controlling all three stones at once.
2. Create three bots controlling one stone each. Use "team spirit" parameter.

### Representation
#### State (*Vector of length 18*)
First six are coordinates of stones for top player (row and column), in case of being out of the game it is 0 0. Second six are coordinates for bottom player.
Next three numbers are states for top player:
1. 2 == stone has totem
2. 1 == stone had totem this turnover
3. 0 == stone did not have totem this turnover
4. -1 == stone is out of the game

Last three numbers are states for bottom player.
#### Action (*Tuple of length 2*)
Actions are coded with:
##### move_to
Enum for where to move. "Out" if there is no plausible move.

##### pass_to
Giving number of stone to whom to pass totem (1,2,3 are stones of top player, 4,5,6 are stones of bottom player).
