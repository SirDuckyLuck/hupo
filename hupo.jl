using Flux
using Flux: onehot
using StatsBase

@enum Move up = 1 right = 2 down = 3 left = 4 out = 5
include("printing.jl")
include("collectingData.jl")
# probability: up+1, up+2, ... up+6, right+1, right+2, ..., right+6m ..., out+6

get_active_stone(state::Array{Int}) = findfirst(view(state, 13:18), 2)
get_active_player(state::Array{Int}) = get_active_stone(state) ∈ (1, 2, 3) ? :top : :bot


function state_beginning(level = :original)
  if level == :original
    return [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
  elseif level == :dummy
    return [3; 2; 1; 2; 3; 2; 3; 2; 5; 2; 3; 2; -1; 2; -1; -1; 0; -1]
  elseif level == :random
    state = zeros(Int, 18)
    stones1 = sample(1:3, rand(1:3), replace = false)
    stones2 = sample(4:6, rand(1:3), replace = false)
    numOfStones = length(stones1) + length(stones2)
    positions = sample([1:4; 6:7; 9:10; 12:15], numOfStones, replace = false)
    k = 0
    for stone ∈ 1:6
      if (stone ∈ stones1) || (stone ∈ stones2)
        k += 1
        (k == 1) && (state[12 + stone] = 2)
        state[stone*2 - 1] = div(positions[k] - 1, 3) + 1
        state[stone*2] = mod(positions[k] - 1, 3) + 1
      else
        state[stone*2 - 1] = 3
        state[stone*2] = 2
        state[12 + stone] = -1
      end
    end

    return state
  end
end


const gX = [Int.(onehot(x, 1:5)) for x in 1:5]
const gY = [Int.(onehot(x, 1:3)) for x in 1:3]
const gA = [Int.(onehot(x, 1:4)) for x in 1:4]

function transformState(state::Array{Int})
  [gX[state[1]]; gX[state[3]]; gX[state[5]]; gX[state[7]]; gX[state[9]]; gX[state[11]];
   gY[state[2]]; gY[state[4]]; gY[state[6]]; gY[state[8]]; gY[state[10]]; gY[state[12]];
   gA[state[13]+2]; gA[state[14]+2]; gA[state[15]+2]; gA[state[16]+2]; gA[state[17]+2]; gA[state[18]+2]]
end


function sample_action(state::Array{Int}, net)
  p = policy(state, net)
  r = rand()
  idx = findfirst(x -> x >= r, cumsum(p))

  move, pass = idx2MovePass(idx)
end


function idx2MovePass(idx::Int)
  move = Move(div(idx - 1, 6) + 1)
  pass = idx % 6 == 0 ? 6 : idx % 6

  move, pass
end


function policy(state::Array{Int}, net)
  p = net(transformState(state)).data .+ 1e-6
  zero_impossible_moves!(p, state)
  p ./= sum(p)

  p
end


function zero_impossible_moves!(p::Array{Float64}, state::Array{Int})
  active_stone = get_active_stone(state)
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

    if check_move_impossibility(new_position, state)
      idx = (Int(move) - 1)*6 + 1 : Int(move)*6
      p[idx] .= 0.
    end
  end

  for pass in 1:6
    if check_pass_impossibility(pass, state)
      idx = pass : 6 : 30
      p[idx] .= 0.
    end
  end

  (any(p[1:24] .> 0.)) && (p[25:30] .= 0.) # if you can do something else than get kicked, do so
end


function check_move_impossibility(new_position::Array{Int}, state::Array{Int})
  # check if there is another stone
  for stone in 1:6
    if new_position == state[stone*2-1:stone*2]
      return true
    end
  end
  # check if middle or out of the board
  if new_position == [3; 2] || new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
    return true
  end

  false
end


function check_pass_impossibility(pass::Int, state::Array{Int})
  if state[12+pass] != 0.
    return true
  end

  false
end


function apply_move!(state::Array{Int}, move::Move)
  active_stone = get_active_stone(state)

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


function apply_pass!(state::Array{Int}, active_stone::Int, pass::Int)
  state[12 + active_stone] != -1 && (state[12 + active_stone] = 1)
  if (active_stone ∈ (1, 2, 3)) != (pass  ∈ (1, 2, 3))
    for s in 13:18
      if state[s] == 1
        state[s] = 0
      end
    end
  end
  state[12 + pass] = 2
end


function check_state(state::Array{Int})
  won = Symbol()

  if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
      won = :top_player_won
  elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
      won = :bottom_player_won
  end

  active_player = get_active_player(state)

  won, active_player
end


function game(net_top, net_bot)
    state = state_beginning()
    active_player = :top
    game_length = 0

    while true
      game_length += 1
      move, pass = active_player == :top ?
                   sample_action(state, net_top) :
                   sample_action(state, net_bot)
      active_stone = apply_move!(state, move)
      apply_pass!(state, active_stone, pass)
      won, active_player = check_state(state)

      if won ∈ [:top_player_won :bottom_player_won]
          return won, game_length
      end
    end
end

# game(net_top, net_bot)
