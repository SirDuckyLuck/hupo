include("hupo.jl")

function loss_moves(states, moves, passes, rewards)
    p_all = net_top_move(states)
    p = sum(Flux.onehotbatch(moves,collect(1:5)) .* p_all, 1)
    Flux.crossentropy(p, rewards)
end

function loss_passes(states, moves, passes, rewards)
    p_all = net_top_pass(states)
    p = sum(Flux.onehotbatch(passes,collect(1:6)) .* p_all, 1)
    Flux.crossentropy(p, rewards)
end


global net_top_move = Chain(
  Dense(18, 100, relu),
  Dense(100, 5),
  softmax)
global net_top_pass = Chain(
  Dense(18, 100, relu),
  Dense(100, 6),
  softmax)
global net_bot_move = Chain(
  Dense(18, 5),
  softmax)
global net_bot_pass = Chain(
  Dense(18, 6),
  softmax)
global numOfEpochs = 1000
global lengthOfBuffer = 500
global r_end = 10.
global r_add = 1.
global discount = 0.5
global opt = SGD(Flux.params(net_top_move), 0.1)

function train_hupo!()
  for epoch in 1:numOfEpochs
    data = collectData!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, r_add, discount)

    Flux.train!(loss_moves, data, opt)

    # safety check --- the probability of going right should rise
    known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
    println(sum(net_top_move(known_state).data[2]))
  end
end

train_hupo!()
game_show(net_top, net_bot)
