
include("hupo.jl")

mutable struct sval_memory_buffer
  N::Int
  k::Int
  states::Matrix{Int}
  rewards::Vector{Float64}
end


function action2idx(move::Move, pass::Int)
  (Int(move) - 1)*6 + pass
end

function sval_memory_buffer(N::Int)
  sval_memory_buffer(N, 0, zeros(Int, 18, N), zeros(N))
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

function sval_policy(state::Array{Int}, sval)
  p = ones(30)
  zero_impossible_moves!(p,state)
  possible_actions = [x for (x,y) in zip(1:30,p) if y==1]

  r = rand()
  if r < epsilon
    i = Int(ceil(rand()*length(possible_actions)))
    idx = possible_actions[i]
  else
    # choose state according to policy
    m = Matrix{Int}(length(transformState(state)),length(possible_actions))
    for i in 1:length(possible_actions)
      new_state = copy(state)
      apply_action!(new_state,idx2MovePass(possible_actions[i]))
      m[:,i] = transformState(new_state)
    end
    svals = sval(m)

    idx = possible_actions[indmax(svals)]
  end

  idx
end


const length_of_game_tolerance = 300
const discount = 0.95
const r_end = 1.
const alpha = 0.1


global sval_top = Dict{UInt64,Float64}()
global count_sval_top = Dict{UInt64,Int}()

global sval_bot = Dict{UInt64,Float64}()
global count_sval_bot = Dict{UInt64,Int}()

global sval_net_top = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
    # Dense(72,1))
global opt_top = ADAM(Flux.params(sval_net_top))

global sval_net_bot = Chain(
    Dense(72, 100, relu),
    Dense(100,1))
global opt_bot = ADAM(Flux.params(sval_net_bot))


function loss_online(state_net::Array{Int}, reward)
  (sval_net_top(state_net)[1,1] - reward)^2
end
function train_net_online!(net,mb,opt)
  Flux.train!(loss_online, [(transformState(mb.states[:,i]),mb.rewards[i]) for i in 1:mb.k], opt)
end

function loss(net)
  (states_net, rewards) -> sum((net(states_net)' - rewards).^2)
end
function train_net!(net,opt,mb)
  m = Matrix{Int}(length(transformState(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = transformState(mb.states[:,i])
  end
  Flux.train!(loss(net), [(m, mb.rewards[1:mb.k])], opt)
end


function loss_bot(states_net::Matrix{Int}, rewards)
  sum((sval_net_bot(states_net)' - rewards).^2)
end
function train_net_bot!(mb)
  m = Matrix{Int}(72,mb.k)
  for i in 1:mb.k
    m[:,i] = transformState(mb.states[:,i])
  end
  Flux.train!(loss_bot, [(m, mb.rewards[1:mb.k])], opt_bot)
end


function loss_top(states_net::Matrix{Int}, rewards)
  sum((sval_net_top(states_net)' - rewards).^2)
end
function train_net_top!(mb)
  m = Matrix{Int}(72,mb.k)
  for i in 1:mb.k
    m[:,i] = transformState(mb.states[:,i])
  end
  Flux.train!(loss_top, [(m, mb.rewards[1:mb.k])], opt_top)
end

const replay_size = 100000
global m_replay = Matrix{Int}(72,replay_size)
global replay_rewards = Vector{Float32}(replay_size)
global replay_k = 0
global replay_full = false

function train_net_top_replay!(mb)
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
  Flux.train!(loss_top, [(m_replay[:,rand_i], replay_rewards[rand_i])], opt_top)
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
  top_wins = 0
  n_games = 0
  sum_game_lenghts = 0

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

    won, game_length = sval_game!(sval_net_top, sval_net_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)

    top_wins += won == :top_player_won ? 1 : 0
    n_games += 1
    sum_game_lenghts += game_length

    # check whether mb contains the right rewards at the right states
    global mb_top = mb_top
    global mb_bot = mb_bot

    if train_top
      train_net!(sval_net_top,opt_top,mb_top)
    end
    if train_bot
      train_net_bot!(mb_bot)
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

    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won = check_state(state)
    active_player = get_active_player(state)

    mb.k +=1
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