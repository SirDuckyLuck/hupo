
#=
monte carlo tree search explained in detial
https://int8.io/monte-carlo-tree-search-beginners-guide/

other sources on MCTS
http://mcts.ai/about/

Alpha zero implementation pseudo-code
https://web.stanford.edu/~surag/posts/alphazero.html
most of this code is actually retyped from this post

other sources on Alpha Zero
https://medium.com/@jonathan_hui/alphago-zero-a-game-changer-14ef6e45eba5
https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef

Article that mentions MCTS looping (but it's not implemented here)
https://arxiv.org/pdf/1805.09218.pdf
=#

using DataStructures: DefaultDict

global visited_states = Set()
global Q = DefaultDict(() -> DefaultDict(0.))
global N = DefaultDict(() -> DefaultDict(0))

function zero_tree!()
  global visited_states = Set()
  global Q = DefaultDict(() -> DefaultDict(0.))
  global N = DefaultDict(() -> DefaultDict(0))
end

const c_param = 1.4

global depth = 0
global cycling_number = 0

global depth_list = []
global rollout_d_list = []
global rollout_d = 0

function mcts_search(p::AbstractMCTSPlayer, state::Array{Int})

  # check to prevent cycling
  global depth += 1
  if depth > 50
    # global cycling_number += 1
    # println(cycling_number)
    # println(length(visited_states))
    return 0.
  end

  # check is terminal node
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    return won == :top_player_won ? 1. : -1.
  end

  if state ∉ visited_states
    push!(visited_states, state)

    return rollout(p, state)
  end

  max_u, best_a = -Inf, -1
  # sum_N = length(N[state]) > 0 ? sum(values(N[state])) : 0
  for a ∈ find_possible_actions(state)
    # get first not expanded node
    if N[state][a] == 0
      best_a = a
      break
    end
    # or calculate the best upper confidence bound node
    # global g_state = state
    # global g_a = a
    # global g_sum_N = sum_N
    sum_N = sum(values(N[state]))
    u = Q[state][a] + c_param * sqrt(log(sum_N) / N[state][a])
    if u>max_u
      max_u, best_a = u, a
    end
  end
  a = best_a

  next_state = copy(state)
  apply_action!(next_state,idx2MovePass(a))
  v = mcts_search(p,next_state)

  this_v = get_active_player(state)==:top ? v : -v
  Q[state][a] = (N[state][a]*Q[state][a] + this_v)/(N[state][a]+1)
  N[state][a] += 1
  return v
end


function sval_sample_action(state::Array{Int}, player::AbstractMCTSPlayer)

  global possible_actions = find_possible_actions(state)

  # zero_tree!()

  for i = 1:length(possible_actions)+1
    # append!(depth_list,depth)
    global depth = 0
    mcts_search(player, copy(state))
  end

  global action_values = [Q[state][a] for a ∈ possible_actions]
  global g_state = state

  # probs = action_values./sum(action_values)
  # r = rand()
  # chosen_i = findfirst(x -> x >= r, cumsum(probs))
  # idx = possible_actions[chosen_i]

  idx = possible_actions[indmax(action_values)]

  move, pass = idx2MovePass(idx)
end


struct BasicMCTSPlayer <: AbstractMCTSPlayer

end

function rollout(p::BasicMCTSPlayer, state)
  global rollout_d = 0
  rollout_state = copy(state)
  while !is_end_state(rollout_state)
    global rollout_d += 1
    actions = find_possible_actions(rollout_state)
    a = actions[rand(1:end)]
    apply_action!(rollout_state,idx2MovePass(a))
  end
  append!(rollout_d_list,rollout_d)

  return check_state(rollout_state)==:top_player_won ? 1 : -1
end

function train!(player::AbstractMCTSPlayer,mb)
  return
end


struct HeuristicMCTSPlayer <: AbstractMCTSPlayer

end

function rollout(p::HeuristicMCTSPlayer, state)
  s = 0.
  for stone = 1:3
    s += state[12+stone]==-1 ? -1 : state[stone*2-1] - 1
  end
  for stone = 4:6
    s -= state[12+stone]==-1 ? -1 : 5 - state[stone*2-1]
  end
  return s / 10
end


struct DictionaryMCTSPlayer <: AbstractMCTSPlayer
  dict
end

function train!(p::DictionaryMCTSPlayer,mb)
  for i = 1:mb.k
    s = mb.states[:,i]
    p.dict[s] = (1. - alpha) * p.dict[s] + alpha * mb.rewards[i]
  end
end

function rollout(p::DictionaryMCTSPlayer, state)
  return p.dict[state]
end


struct NetMCTSPlayer <: AbstractMCTSPlayer
  net
  opt
  transform
end

function loss(net)
  (states_net, rewards) -> sum((net(states_net)' - rewards).^2)
end
function train!(p::NetMCTSPlayer,mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.states[:,i])
  end
  Flux.train!(loss(p.net), [(m, mb.rewards[1:mb.k])], p.opt)
end

function rollout(p::NetMCTSPlayer, state)
  return p.net(p.transform(state)).data[1]
end
