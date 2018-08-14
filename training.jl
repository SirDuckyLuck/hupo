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

function loss(state,action,reward)
    p_all = net_top(state')
    p = sum(Flux.onehotbatch(action,collect(1:30)) .* p_all, 1)
    Flux.crossentropy(p, reward)
end

global numOfEpochs = 10
global lengthOfBuffer = 500

function train_hupo(net_top, net_bot)
  for epoch in 1:numOfEpochs
    memory_buffer = memory_buffer(lengthOfBuffer)
    collectData!(net_top, net_bot, memory_buffer)
    println("Epoch: $(epoch) - average length of game is $(size(st,1)/gamesPerEpoch))")
    println("average reward for top player is $(mean(rt))")
    println("loss is $(loss(st, mt, rt)/gamesPerEpoch)")

    if !isnan(std(rt))
      rt .= (rt .- mean(rt))./std(rt)
    end

    data = Iterators.repeated((st, mt, rt), 1)
    opt = ADAM(Flux.params(net_top))
    Flux.train!(loss, data, opt)
  end
end

train_hupo(net_top, net_bot)
game_show(net_top, net_bot)
