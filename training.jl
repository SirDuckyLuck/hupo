using Flux
include("hupo.jl")

function collectData(numOfGames, net_top, net_bot, exploration)
  states_top = Array{Int}(undef,0,18)
  rewards_top = Vector{Float64}()
  move_top = Vector{Float64}()
  states_bot = Array{Int}(undef,0,18)
  rewards_bot = Vector{Float64}()
  move_bot = Vector{Float64}()

  for i in 1:numOfGames
    st, rt, pt, sb, rb, pb = game(net_top, net_bot, exploration)
    states_top = vcat(states_top,st)
    states_bot = vcat(states_bot,sb)
    rewards_top = vcat(rewards_top,rt)
    rewards_bot = vcat(rewards_bot,rb)
    move_top = vcat(move_top,pt)
    move_bot = vcat(move_bot,pb)
  end

  states_top, rewards_top, move_top, states_bot, rewards_bot, move_bot
end

net_top = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)

net_bot = Chain(
  Dense(18, 100, relu),
  Dense(100, 24),
  softmax)


function loss(state,action,reward)
    p_all = net_top(state')
    p = sum(Flux.onehotbatch(action,collect(1:24)) .* p_all, 1)
    Flux.crossentropy(p, reward)
end


function train_hupo(net_top, net_bot)
  numOfEpochs = 300
  for epoch in 1:numOfEpochs
    st, rt, mt, sb, rb, mb = collectData(100, net_top, net_bot, 1. - epoch/numOfEpochs)
    println("Epoch: $(epoch) - average length of game is $(size(st,1)/100))")
    println("average reward for top player is $(mean(rt))")
    println("loss is $(loss(st, mt, rt))")

    data = Iterators.repeated((st, mt, rt), 1)
    opt = ADAM(Flux.params(net_top))
    Flux.train!(loss, data, opt)
  end
end

train_hupo(net_top, net_bot)
game_show(net_top, net_bot)
