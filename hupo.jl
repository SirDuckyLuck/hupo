include("UnicodeGrids.jl")
include("printing.jl")
include("collectingData.jl")
using .UnicodeGrids
using Statistics
using Flux
using Flux.onehot

@enum Move up = 1 right = 2 down = 3 left = 4

function fill_state_beginning!(state)
  state[:] .= [1; 2; 5; 2; 1]
end

const gX = [Int.(onehot(x, 1:5)) for x in 1:5]
const gY = [Int.(onehot(x, 1:3)) for x in 1:3]
const gA = [Int.(onehot(x, 1:2)) for x in 1:2]

function transformState(state)
  try
  return [gX[state[1]]; gX[state[3]];
    gY[state[2]]; gY[state[4]];
    gA[state[5]]]
  catch
    print_with_color(:green, state)
  end
end


function sample_move(state, net_move)
  active_stone = state[5]
  p = net_move(transformState(state)).data

  stone_position = [state[active_stone*2 - 1];state[active_stone*2]]

  for move in instances(Move) # check moves plausibility except getting out
    new_position = copy(stone_position)
    if move == up
        new_position[1] -=1
    elseif move == right
        new_position[2] +=1
    elseif move == down
        new_position[1] +=1
    elseif move == left
        new_position[2] -=1
    end

    # check for middle
    if new_position == [3; 2]
        p[Int(move)] = 0.
    end
    # check if there is another stone
    for stone in 1:2
        if new_position == state[stone*2-1:stone*2]
            p[Int(move)] = 0.
        end
    end
    # check if out of the board
    if new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
        p[Int(move)] = 0.
    end
  end

  p ./= sum(p)
  r = rand()
  move = Move(findfirst(x -> x >= r, cumsum(p)))
end


function apply_move!(state, move)
  active_stone = state[5]
  if move == up
      state[active_stone*2 - 1] -=1
  elseif move == right
      state[active_stone*2] +=1
  elseif move == down
      state[active_stone*2 - 1] +=1
  elseif move == left
      state[active_stone*2] -=1
  end

  state[5] = active_stone == 1 ? 2 : 1
end


function check_state(state)
  won = Symbol()

  if state[1:2] == [4;2]
      won = :top_player_won
  elseif state[3:4] == [2;2]
      won = :bottom_player_won
  end

  won
end


function game(net_top_move, net_bot_move)
    state = Array{Int}(undef,5)
    fill_state_beginning!(state)
    active_player = :top
    game_length = 0

    while true
      game_length += 1

      move = active_player == :top ? sample_move(state, net_top_move) : sample_move(state, net_bot_move)
      apply_move!(state, move)
      won = check_state(state)
      active_player = active_player == :top ? :bot : :top

      if won ∈ [:top_player_won :bottom_player_won]
          return won, game_length
      end
    end
end


# game(net_top_move, net_bot_move)
