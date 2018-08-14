include("training_helpers.jl")

net_top = Chain(
  Dense(18, 100, relu),
  Dense(100, 100, relu),
  Dense(100, 30),
  softmax)

net_bot = Chain(
  Dense(18, 100, relu),
  Dense(100, 100, relu),
  Dense(100, 30),
  softmax)

function loss(states,actions,rewards)
    p_all = net_top(states)
    p = sum(Flux.onehotbatch(actions,collect(1:30)) .* p_all, 1)
    Flux.crossentropy(p, rewards)
end

global numOfEpochs = 100
global lengthOfBuffer = 500

function train_hupo(net_top, net_bot)
  for epoch in 1:numOfEpochs
    mb = memory_buffer(lengthOfBuffer)
    collectData!(net_top, net_bot, mb)
    println("Epoch: $(epoch)")
    println("Average reward for top player: $(mean(mb.rewards))")
    println("Loss: $(loss(mb.states, mb.actions, mb.rewards)/lengthOfBuffer)")

    if !isnan(std(mb.rewards))
      mb.rewards .= (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
    end

    data = Iterators.repeated((mb.states, mb.actions, mb.rewards), 1)
    opt = ADAM(Flux.params(net_top))
    Flux.train!(loss, data, opt)
  end
end

train_hupo(net_top, net_bot)
game_show(net_top, net_bot)
