include("hupo.jl")

function loss_move(state, move, pass, reward)
    p_all = net_top_move(state)
    p = sum(Flux.onehot(move,collect(1.:5.)) .* p_all)
    -sum(log(p) * reward)
end

function loss_pass(state, move, pass, reward)
    p_all = net_top_pass(state)
    p = sum(Flux.onehot(pass,collect(1.:6.)) .* p_all)
    -sum(log(p) * reward)
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
global lengthOfBuffer = 300
global r_end = 1.
global r_add = 1.
global discount = 0.75
global opt_move = SGD(Flux.params(net_top_move), 0.0001)
global opt_pass = SGD(Flux.params(net_top_pass), 0.0001)

function train_hupo!()
  for epoch in 1:numOfEpochs
    data = collectData!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, r_add, discount)

    for i in 1:lengthOfBuffer
      Flux.train!(loss_move, zip(data[1][:,i],data[2][i],data[3][i],data[4][i]), opt_move)
      Flux.train!(loss_pass, zip(data[1][:,i],data[2][i],data[3][i],data[4][i]), opt_pass)
    end

    #check against random net
    if (epoch % 100 == 0)
      dummy_games = [game(net_top_move, net_top_pass, net_bot_move, net_bot_pass) for i in 1:100]
      net_top_wins = sum(dummy_games.=="top player won")
      println("Epoch: $(epoch), net_top won $net_top_wins against random net")
      # safety check --- the probability of going right should rise
      known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      println(sum(net_top_move(known_state).data[2]))
    end
  end
end

train_hupo!()
game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
