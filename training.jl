include("hupo.jl")

const numOfEpochs = 10000
const lengthOfBuffer = 500
const r_end = 1.
const r_add = 1.
const discount = 0.95
const learning_rate = 1e-5
const length_of_game_tolerance = 50

const net_top_move = Chain(
    Dense(18, 100, relu),
    Dense(100, 100, relu),
    Dense(100, 5),
    softmax)
const net_top_pass = Chain(
    Dense(18, 100, relu),
    Dense(100, 100, relu),
    Dense(100, 6),
    softmax)
const net_bot_move = Chain(
    Dense(18, 5),
    softmax)
const net_bot_pass = Chain(
    Dense(18, 6),
    softmax)


function loss_move(state, move, reward)
    p = net_top_move(state)[move]
    -sum(log(p) * reward)
end

function loss_pass(state, pass, reward)
    p = net_top_pass(state)[pass]
    -sum(log(p) * reward)
end


function train_hupo!()
  opt_move = SGD(Flux.params(net_top_move), learning_rate)
  opt_pass = SGD(Flux.params(net_top_pass), learning_rate)

  for epoch in 1:numOfEpochs
    data = collectData!(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, r_add, discount, length_of_game_tolerance)

    for i in 1:lengthOfBuffer
      Flux.train!(loss_move, zip(data[1][:,i],data[2][i],data[4][i]), opt_move)
      Flux.train!(loss_pass, zip(data[1][:,i],data[3][i],data[4][i]), opt_pass)
    end

    if (epoch % 100 == 0)
      known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      n1 = net_top_move(known_state).data
      println("Safety check moving: $(n1) which is $(n1[2]/(n1[1] + n1[2])*100) %")
      known_state = [1; 1; 1; 2; 1; 3; 5; 1; 4; 1; 5; 2; 1; 2; 1; 0; 0; 0]
      n2 = net_top_pass(known_state).data
      println("Safety check passing: $(n2) which is $(n2[4]/(n2[4] + n2[5] + n2[6])*100) %")
    end

    #check against random net
    if (epoch % 1000 == 0)
      dummy_games = [game(net_top_move, net_top_pass, net_bot_move, net_bot_pass) for i in 1:100]
      net_top_wins = sum(dummy_games.==:top_player_won)
      println("Epoch: $(epoch), net_top won $net_top_wins against random net")
      # safety check --- the probability of going right should rise
      # known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      # println("Safety check: $(net_top_move(known_state).data[2])")
    end
  end
end


train_hupo!()
game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
