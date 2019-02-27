
include("statePlayer.jl")

struct HeuristicPlayer <: AbstractStatePlayer
  top_or_bot
end

global heu_epsilon = 0.1
function get_epsilon(p::HeuristicPlayer)
  return heu_epsilon
end

function get_sval(player::HeuristicPlayer,state)
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    this_player_won(player.top_or_bot,won) ? 1. : -1.
  else
    player.top_or_bot == :bot && (state = invert_state(state))
    s = 0
    for stone in 1:3
      s += state[12+stone]==-1 ? -1 : state[stone*2-1] - 1
    end
    s / 10
  end
end


function actions_svals(state::Array{Int}, player::HeuristicPlayer)
  possible_actions = find_possible_actions(state)

  svals = map(possible_actions) do i
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(i))
    if get_active_player(state) == get_active_player(new_state)
      as,vs = actions_svals(new_state,player)
      maximum(vs)
    else
      get_sval(player,new_state)
    end
  end

  possible_actions,svals
end


function train!(p::HeuristicPlayer,mb)
  return 0
end




struct Heuristic2Player <: AbstractStatePlayer
  top_or_bot
end

# global get_sval_count = 0

function get_sval(player::Heuristic2Player,state)
  # global get_sval_count += 1
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    this_player_won(player.top_or_bot,won) ? 1. : -1.
  else
    player.top_or_bot == :bot && (state = invert_state(state))
    s = 0
    for stone in 1:3
      s += state[12+stone]==-1 ? -1 : state[stone*2-1] - 1
    end
    for stone in 4:6
      s -= state[12+stone]==-1 ? -1 : 5 - state[stone*2-1]
    end
    s / 10
  end
end


function actions_svals(state::Array{Int}, player::Heuristic2Player)
  possible_actions = find_possible_actions(state)

  svals = map(possible_actions) do i
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(i))
    if check_state(new_state) ∈ (:top_player_won, :bottom_player_won)
      get_sval(player,new_state)
    elseif player.top_or_bot == get_active_player(new_state)
      as,vs = actions_svals(new_state,player)
      maximum(vs)
    else
      as, vs = enemy_actions_svals(new_state,player)
      minimum(vs)
    end
  end

  possible_actions,svals
end

function enemy_actions_svals(state::Array{Int}, player::Heuristic2Player)
  possible_actions = find_possible_actions(state)

  svals = map(possible_actions) do i
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(i))
    if check_state(new_state) ∈ (:top_player_won, :bottom_player_won)
      get_sval(player,new_state)
    elseif player.top_or_bot != get_active_player(new_state)
      as,vs = enemy_actions_svals(new_state,player)
      minimum(vs)
    else
      get_sval(player,new_state)
    end
  end

  possible_actions,svals
end


function train!(p::Heuristic2Player,mb)
  return
end

# global sample_action_count = 0

function sval_sample_action(state::Array{Int}, player::Heuristic2Player)
  possible_actions, svals = actions_svals(state, player)

  # global sample_action_count += 1

  r = rand()
  if r < epsilon
    idx = possible_actions[rand(1:end)]
  else
    max = maximum(svals)
    max_is = find(svals .== max)
    rand_max = max_is[rand(1:end)]
    idx = possible_actions[rand_max]
  end

  move, pass = idx2MovePass(idx)
end
