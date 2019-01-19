

struct ActionPlayer <: AbstractActorPlayer
  net
  opt
  transform
end




struct ACPlayer <: AbstractActorPlayer
  action_net
  action_opt
  critic_net
  critic_opt
  transform
end



function sval_sample_action(state::Array{Int}, player::AbstractActorPlayer)
  probs = action_probs(state,player)

  r = rand()
  idx = findfirst(x -> x >= r, cumsum(probs))

  move, pass = idx2MovePass(idx)
end



function loss_action(net)
  (states_net, actions, rewards) -> Flux.crossentropy(net(states_net)[Flux.onehotbatch(actions, 1:30)] .+ 1e-8, rewards)
end
function train!(p::ActionPlayer,mb)
  m = Matrix{Int}(length(p.transform(rand_state())),mb.k)
  for i in 1:mb.k
    m[:,i] = p.transform(mb.start_states[:,i])
  end

  # try normalizing rewards
  mb.rewards = (mb.rewards .- mean(mb.rewards))./std(mb.rewards)

  Flux.train!(loss_action(p.net), [(m, mb.actions[1:mb.k], mb.rewards[1:mb.k])], p.opt)
end

function action_probs(state::Array{Int}, player::ActionPlayer)
  p = player.net(player.transform(state)).data .+ 1e-6
  zero_impossible_moves!(p, state)
  p ./= sum(p)

  p
end



function train!(p::ACPlayer,mb)
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

function action_probs(state::Array{Int}, player::ACPlayer)
  p = player.action_net(player.transform(state)).data .+ 1e-6
  zero_impossible_moves!(p, state)
  p ./= sum(p)

  p
end
