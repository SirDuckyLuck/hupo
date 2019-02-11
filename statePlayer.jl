

struct NetPlayer <: AbstractStatePlayer
  net
  opt
  transform
end

function invert_player(p::NetPlayer)
  new_net = deepcopy(p.net)
  new_transform = state -> p.transform(invert_state(state))
  NetPlayer(new_net,ADAM(Flux.params(new_net)),new_transform)
end


struct ImprovedNetPlayer <: AbstractStatePlayer
  net
  opt
  transform
  top_or_bot
end

function get_sval(player::ImprovedNetPlayer,state)
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    this_player_won(player.top_or_bot,won) ? 1. : -1.
  else
    player.net(player.transform(state)).data[1]
  end
end


struct ImprovedMCPlayer <: AbstractStatePlayer
  dict
  top_or_bot
end

function get_sval(player::ImprovedMCPlayer,state)
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    this_player_won(player.top_or_bot,won) ? 1. : -1.
  else
    get(player.dict,state2hash(state),0.)
  end
end

struct LookaheadMCPlayer <: AbstractStatePlayer
  dict
  top_or_bot
end

function get_sval(player::LookaheadMCPlayer, state, level=1)
  won = check_state(state)
  if won ∈ (:top_player_won, :bottom_player_won)
    this_player_won(player.top_or_bot,won) ? 1. : -1.
  elseif level==3
    get(player.dict,state2hash(state),0.)
  else
    possible_actions = find_possible_actions(state)

    svals = map(possible_actions) do i
      new_state = copy(state)
      apply_action!(new_state,idx2MovePass(i))
      get_sval(player,new_state, level+1)
    end

    if get_active_player(state) == player.top_or_bot
      maximum(svals)
    else
      minimum(svals)
    end
  end
end

function state2hash(state::Array{Int})
  # here implement transforming state into a UInt64 integer
  return hash(state)
end

# struct StatePlayer <: AbstractStatePlayer
#   dict
# end
#
# function get(p::StatePlayer,key,value)
#   get(p.dict,key,value)
# end
#
# function setindex!(p::StatePlayer,value,key)
#   p.dict[key] = value
# end

const epsilon = 0.1
const alpha = 0.1

function sval_sample_action(state::Array{Int}, player::Union{AbstractStatePlayer,Dict})
  possible_actions, svals = actions_svals(state, player)

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


function actions_svals(state::Array{Int}, sval::Dict)
  possible_actions = find_possible_actions(state)

  svals = map(possible_actions) do i
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(i))
    get(sval,state2hash(new_state),0)
  end

  possible_actions,svals
end

function actions_svals(state::Array{Int}, player::Union{ImprovedMCPlayer,ImprovedNetPlayer})
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

function actions_svals(state::Array{Int}, player::LookaheadMCPlayer)
  possible_actions = find_possible_actions(state)

  svals = map(possible_actions) do i
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(i))
    get_sval(player,new_state)
  end

  possible_actions,svals
end

function actions_svals(state::Array{Int}, player::NetPlayer)
  possible_actions = find_possible_actions(state)

  m = Matrix{Int}(length(player.transform(state)),length(possible_actions))
  for i in 1:length(possible_actions)
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(possible_actions[i]))
    m[:,i] = player.transform(new_state)
  end
  svals = vec(player.net(m).data)

  possible_actions,svals
end



function loss(net)
  (states_net, rewards) -> sum((net(states_net)' - rewards).^2)
end
function train!(p::Union{NetPlayer,ImprovedNetPlayer},mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.states[:,i])
  end
  Flux.train!(loss(p.net), [(m, mb.rewards[1:mb.k])], p.opt)
end


function train!(p::ImprovedMCPlayer,mb)
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    prev_mean = get_sval(p,mb.states[:,i])
    p.dict[s] = prev_mean + alpha * (mb.rewards[i] - prev_mean)
  end
end


function train!(p::LookaheadMCPlayer,mb)
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    prev_mean = get_sval(p,mb.states[:,i])
    p.dict[s] = prev_mean + alpha * (mb.rewards[i] - prev_mean)
    # update also the start state values
    ss = state2hash(mb.start_states[:,i])
    p.dict[ss] = (1-alpha) * get_sval(p,mb.start_states[:,i]) + alpha * mb.rewards[i]
  end
end


const replay_size = 100000
global m_replay = Matrix{Int}(72,replay_size)
global replay_rewards = Vector{Float32}(replay_size)
global replay_k = 0
global replay_full = false

function train_net_replay!(p::NetPlayer,mb)
  # store replay values
  global replay_k
  for i = 1:mb.k
    if replay_k + i > replay_size
      replay_k -= replay_size
      replay_full = true
    end
    m_replay[:,replay_k + i] = p.transform(mb.states[:,i])
    replay_rewards[replay_k + i] = mb.rewards[i]
  end
  replay_k += mb.k
  # draw random indices
  batch_size = mb.k
  if replay_full
    rand_i = rand(1:replay_size,batch_size)
  else
    rand_i = rand(1:replay_k,batch_size)
  end
  # train on these
  Flux.train!(loss(p.net), [(m_replay[:,rand_i], replay_rewards[rand_i])], p.opt)
end


function train_temporal_difference!(sval,mb)
  for i = mb.k-1:-1:1
  # for i = 1:mb.k-1
    s = state2hash(mb.states[:,i])
    prev_sval = get(sval,s,0.)
    next_sval = get(sval,state2hash(mb.states[:,i+1]),0.)
    if i==mb.k-1
      td_target = mb.rewards[mb.k] + discount*next_sval
    else
      td_target = discount * next_sval
    end
    sval[s] = prev_sval + alpha * (td_target - prev_sval)
  end
end

const lambda = 0.9

function train_td_lambda!(sval,mb)
  eligibility_top = Dict{UInt64,Float32}()
  for i = 1:mb.k-1
    s = state2hash(mb.states[:,i])
    # compute td delta
    prev_sval = get(sval,s,0.)
    next_sval = get(sval,state2hash(mb.states[:,i+1]),0.)
    if i==mb.k-1
      td_delta = mb.rewards[mb.k] + discount*next_sval - prev_sval
    else
      td_delta = discount * next_sval - prev_sval
    end
    # add one to eligibility trace
    # eligibility_top[s] = get(eligibility_top,s,0.) + 1
    eligibility_top[s] = 1
    # add the delta eligibility to the state values
    if td_delta!=0
      for e_state in keys(eligibility_top)
        sval[e_state] = get(sval,e_state,0.) + alpha * td_delta * eligibility_top[e_state]
        eligibility_top[e_state] *= discount * lambda
      end
    end
  end
end

function train!(sval::Dict,mb)
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    prev_mean = get(sval,s,0.)
    sval[s] = prev_mean + alpha * (mb.rewards[i] - prev_mean)
  end
end
