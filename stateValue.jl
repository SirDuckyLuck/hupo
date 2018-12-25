
include("hupo.jl")

mutable struct sval_memory_buffer
  N::Int
  k::Int
  states::Matrix{Int}
  rewards::Vector{Float64}
  start_states::Matrix{Int}
  actions::Vector{Int}
end

struct NetPlayer
  net
  opt
  transform
end

struct ActionPlayer
  net
  opt
  transform
end

struct ACPlayer
  action_net
  action_opt
  critic_net
  critic_opt
  transform
end

function action2idx(move::Move, pass::Int)
  (Int(move) - 1)*6 + pass
end

function sval_memory_buffer(N::Int)
  sval_memory_buffer(N, 0, zeros(Int, 18, N), zeros(N), zeros(Int, 18, N), zeros(N))
end

function state2hash(state::Array{Int})
  # here implement transforming state into a UInt64 integer
  return hash(state)
end

const epsilon = 0.1

function sval_sample_action(state::Array{Int}, sval)
  # find possible idx for movePass and corresponding next states
  if get_active_player(state) == :top
    idx = sval_policy(state,sval)
  else
    # idx = sval_policy(invert_state(state),sval)
    # p = zeros(30)
    # p[idx] = 1
    # p[1:6], p[13:18] = p[18:-1:13], p[6:-1:1]
    # p[7:12], p[19:24] = p[24:-1:19], p[12:-1:7]
    # p[25:30] = p[30:-1:25]
    # idx = indmax(p)
    idx = sval_policy(state,sval)
  end

  move, pass = idx2MovePass(idx)
end

function sval_policy(state::Array{Int}, sval::Dict)
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

  idx
end

function sval_policy(state::Array{Int}, player::NetPlayer)
  p = ones(30)
  zero_impossible_moves!(p,state)
  possible_actions = [x for (x,y) in zip(1:30,p) if y==1]

  r = rand()
  if r < epsilon
    i = Int(ceil(rand()*length(possible_actions)))
    idx = possible_actions[i]
  else
    # choose state according to policy
    m = Matrix{Int}(length(player.transform(state)),length(possible_actions))
    for i in 1:length(possible_actions)
      new_state = copy(state)
      apply_action!(new_state,idx2MovePass(possible_actions[i]))
      m[:,i] = player.transform(new_state)
    end
    svals = player.net(m)

    idx = possible_actions[indmax(svals)]
  end

  idx
end

function sval_policy(state::Array{Int}, player::ActionPlayer)
  p = player.net(player.transform(state)).data .+ 1e-6
  zero_impossible_moves!(p, state)
  p ./= sum(p)

  r = rand()
  idx = findfirst(x -> x >= r, cumsum(p))
end

function sval_policy(state::Array{Int}, player::ACPlayer)
  p = player.action_net(player.transform(state)).data .+ 1e-6
  zero_impossible_moves!(p, state)
  p ./= sum(p)

  r = rand()
  idx = findfirst(x -> x >= r, cumsum(p))
end

const length_of_game_tolerance = 300
const discount = 0.95
const r_end = 1.
const alpha = 0.1

global action_net_top = Chain(
    Dense(72, 100, relu),
    Dense(100,30),
    softmax)

global action_player_top = ActionPlayer(action_net_top,ADAM(Flux.params(action_net_top)),transformState)

function loss_action(net)
  (states_net, actions, rewards) -> Flux.crossentropy(net(states_net)[Flux.onehotbatch(actions, 1:30)] .+ 1e-8, rewards)
end
function train_action_net!(p::ActionPlayer,mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.start_states[:,i])
  end

  # try normalizing rewards
  mb.rewards = (mb.rewards .- mean(mb.rewards))./std(mb.rewards)

  Flux.train!(loss_action(p.net), [(m, mb.actions[1:mb.k], mb.rewards[1:mb.k])], p.opt)
end


global sval_net_top = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
    # Dense(72,1))
global player_top = NetPlayer(sval_net_top,ADAM(Flux.params(sval_net_top)),transformState)

global sval_net_bot = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
global player_bot = NetPlayer(sval_net_bot,ADAM(Flux.params(sval_net_bot)),transformState)

function loss(net)
  (states_net, rewards) -> sum((net(states_net)' - rewards).^2)
end
function train_net!(p::NetPlayer,mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.states[:,i])
  end
  Flux.train!(loss(p.net), [(m, mb.rewards[1:mb.k])], p.opt)
end


global ac_player_top = ACPlayer(action_net_top,ADAM(Flux.params(action_net_top)),sval_net_top,ADAM(Flux.params(sval_net_top)),transformState)

function train_ac_player!(p::ACPlayer,mb)
  # train the critic net
  m1 = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m1[:,i] = p.transform(mb.states[:,i])
  end
  Flux.train!(loss(p.critic_net), [(m1, mb.rewards[1:mb.k])], p.critic_opt)

  # train the action net
  m2 = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m2[:,i] = p.transform(mb.start_states[:,i])
  end
  # find the estimated values for these states
  value_estimates = p.critic_net(m1)'
  # ??? how come we have start_states and states, which ones do we take now for this
  # the mb.rewards[1:mb.k] must be replaced in the next line
  Flux.train!(loss_action(p.action_net), [(m2, mb.actions[1:mb.k], value_estimates)], p.action_opt)
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

global sval_top = Dict{UInt64,Float64}()
global count_sval_top = Dict{UInt64,Int}()

global sval_bot = Dict{UInt64,Float64}()
global count_sval_bot = Dict{UInt64,Int}()

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

function train_monte_carlo!(sval,count_sval,mb)
  # this is every visit monte carlo policy evaluation
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    # update the monte carlo counter
    count_sval[s] = get(count_sval,s, 0) + 1
    # update the monte carlo mean return
    prev_mean = get(sval,s,0.)
    sval[s] = prev_mean + (mb.rewards[i] - prev_mean) / count_sval[s]
  end
end

function train_monte_carlo_online!(sval,mb)
  for i = 1:mb.k
    s = state2hash(mb.states[:,i])
    prev_mean = get(sval,s,0.)
    sval[s] = prev_mean + alpha * (mb.rewards[i] - prev_mean)
  end
end

function computeStateValue!(;n_epochs = 1000, train_bot = false, train_top = false, level = :original)

  epoch = 0
  while epoch <= n_epochs

    if epoch % 1000 == 0
      top_wins = 0
      n_games = 0
      sum_game_lenghts = 0
    end
    epoch += 1

    mb_top = sval_memory_buffer(length_of_game_tolerance)
    mb_bot = sval_memory_buffer(length_of_game_tolerance)

    won, game_length = sval_game!(ac_player_top, player_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)

    top_wins += won == :top_player_won ? 1 : 0
    n_games += 1
    sum_game_lenghts += game_length

    # check whether mb contains the right rewards at the right states
    global mb_top = mb_top
    global mb_bot = mb_bot

    if train_top
      train_ac_player!(ac_player_top, mb_top)
    end
    if train_bot
      train_net!(player_bot, mb_bot)
    end

    if epoch % 1000 == 0
      println("Epoch: $(epoch)")
      top_win = round(top_wins / n_games * 100,1)
      avg_length = round(sum_game_lenghts / n_games,1)
      println("top won $top_win % against bot in $avg_length rounds")
    end
  end

end


function sval_game!(sval_top, sval_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)
  state = state_beginning(level)
  active_player = :top
  won = Symbol()
  game_length = 0

  while true
    game_length += 1
    move, pass = active_player == :top ?
                 sval_sample_action(state, sval_top) :
                 sval_sample_action(state, sval_bot)
    if (active_player == :top)
      mb = mb_top
    else
      mb = mb_bot
    end

    mb.k +=1
    mb.start_states[:,mb.k] = state
    mb.actions[mb.k] = action2idx(move, pass)

    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won = check_state(state)
    active_player = get_active_player(state)

    mb.states[:,mb.k] = state

    if game_length > length_of_game_tolerance
      won = :bottom_player_won
    end

    if won ∈ (:top_player_won, :bottom_player_won)
      mb = mb_top
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - 1):-1:0)) .* r_end
        if won == :top_player_won
            mb.rewards[1:mb.k] .+= reward
        else
            mb.rewards[1:mb.k] .-= reward
        end
      end
      mb = mb_bot
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - 1):-1:0)) .* r_end
        if won == :bottom_player_won
            mb.rewards[1:mb.k] .+= reward
        else
            mb.rewards[1:mb.k] .-= reward
        end
      end
      return won, game_length
    end

  end
end
