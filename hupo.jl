using Flux
using Flux: onehot
using Distributions: Categorical

@enum Move up = 1 right = 2 down = 3 left = 4 out = 5
include("printing.jl")
include("collectingData.jl")
# probability: up+1, up+2, ... up+6, right+1, right+2, ..., right+6m ..., out+6

get_active_stone(state::Array{Int}) = findfirst(view(state, 13:18), 2)
get_active_player(state::Array{Int}) = get_active_stone(state) < 4 ? :top : :bot


function state_beginning(level = :original)
  if level == :original
    return [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
  elseif level == :dummy
    return [3; 2; 1; 2; 3; 2; 3; 2; 5; 2; 3; 2; -1; 2; -1; -1; 0; -1]
  elseif level == :random
    return rand_state()
  end
end


function rand_state()
  state = [3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, -1, -1, -1, -1, -1, -1] # all dead
  positions = shuffle([(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,3), (4,1), (4,3), (5,1), (5,2), (5,3)])

  n_our_live_stones = rand(Categorical([0.1, 0.2, 0.7]))
  our_live_stones = shuffle(1:3)[1:n_our_live_stones]
  for i ∈ our_live_stones
    state[i + 12] = rand(0:1)
    state[2i - 1], state[2i] = pop!(positions)
  end
  state[12 + our_live_stones[1]] = 2 # active stone

  n_their_live_stones = rand(Categorical([0.1, 0.2, 0.7]))
  their_live_stones = shuffle(4:6)[1:n_their_live_stones]
  for i ∈ their_live_stones
    state[i + 12] = 0
    state[2i - 1], state[2i] = pop!(positions)
  end

  state
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
  if get_active_player(state) == :top
    p = net(transformState(state)).data .+ 1e-6
    zero_impossible_moves!(p, state)
    p ./= sum(p)
    return p
  else
    p = net(transformState(state)).data .+ 1e-6
    zero_impossible_moves!(p, state)
    # state = invert_state(state)
    # p = net(transformState(state)).data .+ 1e-6
    # zero_impossible_moves!(p, state)
    # p[1:6], p[13:18] = p[18:-1:13], p[6:-1:1]
    # p[7:12], p[19:24] = p[24:-1:19], p[12:-1:7]
    # p[25:30] = p[30:-1:25]
    p ./= sum(p)
    return p
  end
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


function find_possible_actions(state::Array{Int})
  p = ones(30)
  zero_impossible_moves!(p,state)
  find(p.==1)
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

  won
end

function is_end_state(state::Array{Int})
  won = check_state(state)
  return won ∈ (:top_player_won, :bottom_player_won)
end

function this_player_won(player,won)
  if (player==:top) && (won==:top_player_won)
    true
  elseif (player==:bot) && (won==:bottom_player_won)
    true
  else
    false
  end
end


function apply_action(state, action)
  move, pass = action
  new_state = copy(state)
  active_stone = apply_move!(new_state, move)
  apply_pass!(new_state, active_stone, pass)
  return new_state
end


net_player(net) = (state -> sample_action(state, net))


function invert_state(state)
  state = copy(state)
  state[13:18] = state[18:-1:13]
  for i ∈ 1:2:11
    state[i] = 6 - state[i]
  end
  for i ∈ 2:2:12
    state[i] = 4 - state[i]
  end
  state[1:2], state[11:12] = state[11:12], state[1:2]
  state[3:4], state[9:10] = state[9:10], state[3:4]
  state[5:6], state[7:8] = state[7:8], state[5:6]
  state
end


function game(player_top, player_bottom; level = :original)
  state = state_beginning(level)
  active_player = player_top
  game_length = 0

  while true
    game_length += 1
    action = active_player(state)
    state = apply_action(state, action)
    won = check_state(state)
    if won ∈ [:top_player_won :bottom_player_won]
      return won, game_length
    end
    active_player = get_active_stone(state) < 4 ? player_top : player_bottom
  end
end

# game(net_top, net_bot)
