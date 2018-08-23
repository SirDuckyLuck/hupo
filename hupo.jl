include("UnicodeGrids.jl")
include("printing.jl")
include("collectingData.jl")
using .UnicodeGrids
using Statistics
using Flux
using Flux: onehot

@enum Move up = 1 right = 2 down = 3 left = 4 out = 5


get_active_stone(state) = findfirst(view(state, 13:18), 2)
get_active_player(active_stone) = active_stone ∈ (1, 2, 3) ? :top : :bot

function fill_state_beginning!(state)
  state[:] .= [3; 2; 1; 2; 3; 2; 3; 2; 5; 2; 3; 2; -1; 2; -1; -1; 0; -1]
end

const gX = [Int.(onehot(x, 1:5)) for x in 1:5]
const gY = [Int.(onehot(x, 1:3)) for x in 1:3]
const gA = [Int.(onehot(x, 1:4)) for x in 1:4]

function transformState(state)
  try
  return [gX[state[1]]; gX[state[3]]; gX[state[5]]; gX[state[7]]; gX[state[9]]; gX[state[11]];
    gY[state[2]]; gY[state[4]]; gY[state[6]]; gY[state[8]]; gY[state[10]]; gY[state[12]];
    gA[state[13]+2]; gA[state[14]+2]; gA[state[15]+2]; gA[state[16]+2]; gA[state[17]+2]; gA[state[18]+2]]
  catch
    print_with_color(:green, state)
  end
end


function sample_move(state, active_stone, net_move)
  p = net_move(transformState(state)).data .+ 1e-6

  stone_position = [state[active_stone*2 - 1];state[active_stone*2]]

  for move in instances(Move)[1:4] # check moves plausibility except getting out
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
    for stone in 1:6
        if new_position == state[stone*2-1:stone*2]
            p[Int(move)] = 0.
        end
    end
    # check if out of the board
    if new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
        p[Int(move)] = 0.
    end
  end

  (all(p .== 0.)) && (return out) # if you can do something else than get kicked, do
  p ./= sum(p)
  r = rand()
  move = Move(findfirst(x -> x >= r, cumsum(p)))
end


function sample_pass(state, active_stone, net_pass)
  p = net_pass(transformState(state)).data .+ 1e-6

  for pass in 1:6 # check passing plausibility
      if state[12+pass] != 0.
          p[pass] = 0.
      end
  end

  p ./= sum(p)

  r = rand()
  pass = findfirst(x -> x >= r, cumsum(p))
end


function apply_move!(state, active_stone, move)
  if move == up
      state[active_stone*2 - 1] -=1
  elseif move == right
      state[active_stone*2] +=1
  elseif move == down
      state[active_stone*2 - 1] +=1
  elseif move == left
      state[active_stone*2] -=1
  elseif move == out
        state[(active_stone*2-1):(active_stone*2)] = [3;2]
        state[12+active_stone] = -1
  end

  active_stone
end

function apply_pass!(state, active_stone, pass)
  state[12 + active_stone] != -1 && (state[12 + active_stone] = 1)
  if get_active_player(active_stone) != get_active_player(pass)
    for s in 13:18
      if state[s] == 1
        state[s] = 0
      end
    end
  end
  state[12 + pass] = 2

  pass
end


function check_state(state, active_stone)
  won = Symbol()

  if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
      won = :top_player_won
  elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
      won = :bottom_player_won
  end

  active_player = get_active_player(active_stone)

  won, active_player
end


function game(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = :top
    active_stone = 2
    game_length = 0

    while true
      game_length += 1

      move = active_player == :top ? sample_move(state, active_stone, net_top_move) : sample_move(state, active_stone, net_bot_move)
      active_stone = apply_move!(state, active_stone, move)
      pass = active_player == :top ? sample_pass(state, active_stone, net_top_pass) : sample_pass(state, active_stone, net_bot_pass)
      active_stone = apply_pass!(state, active_stone, pass)
      won, active_player = check_state(state, active_stone)

      if won ∈ [:top_player_won :bottom_player_won]
          return won, game_length
      end
    end
end


# game(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
