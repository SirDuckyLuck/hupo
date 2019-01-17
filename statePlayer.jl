
abstract type AbstractStatePlayer <: AbstractPlayer end

struct NetPlayer <: AbstractStatePlayer
  net
  opt
  transform
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


function sval_sample_action(state::Array{Int}, player::AbstractStatePlayer)
  probs = action_probs(state, player)

  r = rand()
  if r < epsilon
    possible_actions = [x for (x,y) in zip(1:30,probs) if y>0]
    i = Int(ceil(rand()*length(possible_actions)))
    idx = possible_actions[i]
  else
    idx = indmax(probs)
  end

  move, pass = idx2MovePass(idx)
end

function sval_sample_action(state::Array{Int}, sval::Dict)
  p = ones(30)
  zero_impossible_moves!(p,state)
  possible_actions = [x for (x,y) in zip(1:30,p) if y==1]

  r = rand()
  if r < epsilon
    i = Int(ceil(rand()*length(possible_actions)))
    idx = possible_actions[i]
  else
    # choose state according to policy
    svals = map(possible_actions) do i
      new_state = copy(state)
      apply_action!(new_state,idx2MovePass(i))
      get(sval,state2hash(new_state),0)
    end
    idx = possible_actions[indmax(svals)]
  end

  move, pass = idx2MovePass(idx)
end




function loss(net)
  (states_net, rewards) -> sum((net(states_net)' - rewards).^2)
end
function train!(p::NetPlayer,mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.states[:,i])
  end
  Flux.train!(loss(p.net), [(m, mb.rewards[1:mb.k])], p.opt)
end

function action_probs(state::Array{Int}, player::NetPlayer)
  p = ones(30)
  zero_impossible_moves!(p,state)
  possible_actions = [x for (x,y) in zip(1:30,p) if y==1]

  # calculate state values according to the net
  m = Matrix{Int}(length(player.transform(state)),length(possible_actions))
  for i in 1:length(possible_actions)
    new_state = copy(state)
    apply_action!(new_state,idx2MovePass(possible_actions[i]))
    m[:,i] = player.transform(new_state)
  end
  p[possible_actions] = player.net(m).data

  p ./= sum(p)
  p
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
    m_replay[:,replay_k + i] = transformState(mb.states[:,i])
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

function train_monte_carlo_online!(sval,mb)
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    prev_mean = get(sval,s,0.)
    sval[s] = prev_mean + alpha * (mb.rewards[i] - prev_mean)
  end
end
