include("hupo.jl")

net_top = Chain(
  Dense(18, 500, relu),
  Dense(500, 500, relu),
  Dense(500, 30),
  softmax)

net_bot = Chain(
  Dense(18, 30),
  softmax)

function loss(states,actions,rewards)
    p_all = net_top(states)
    p = sum(Flux.onehotbatch(actions,collect(1:30)) .* p_all, 1)
    Flux.crossentropy(p, rewards)
end

global numOfEpochs = 1000
global lengthOfBuffer = 300
global opt = ADAM(Flux.params(net_top))

function train_hupo(net_top, net_bot)
  for epoch in 1:numOfEpochs
    mb = memory_buffer(lengthOfBuffer)
    collectData!(net_top, net_bot, mb)

    if !isnan(std(mb.rewards))
      mb.rewards .= (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
    end

    data = Iterators.repeated((mb.states, mb.actions, mb.rewards), 1)

    Flux.train!(loss, data, opt)

    if (epoch % 100 == 0)
      deciding_games = [game(net_top, net_bot) for i in 1:100]
      top_wins = sum(deciding_games.=="top player won")
      println("Epoch: $(epoch), top won $top_wins")
    end
  end
end

train_hupo(net_top, net_bot)
# game_show(net_top, net_bot)
