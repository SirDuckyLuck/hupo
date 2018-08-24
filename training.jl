include("hupo.jl")

numOfEpochs = 10 ^ 6
const lengthOfBuffer = 300
const r_end = 1.
const discount = 0.9
const learning_rate = 1e-4
const length_of_game_tolerance = 1001
const minP = 0.010
const maxP = 1-minP

const net_top_move = Chain(
    Dense(18, 200, relu),
    Dense(200, 4),
    softmax)
const net_bot_move = Chain(
    Dense(18, 4),
    softmax)

function loss_move(state, move, reward)
    p = net_top_move(state)[move]
    if ((reward < 0.) && (minP < p)) || ((reward > 0.) && (maxP > p))
      -log(p) * reward
    else
      0p
    end
end

function signal_handler(sig::Cint)::Void
    global numOfEpochs = 0
    return
end
signal_handler_c = cfunction(signal_handler, Void, (Cint,))
ccall(:signal, Cint, (Cint, Ptr{Void}), 2, signal_handler_c)


function train_hupo!()
  opt_move = SGD(Flux.params(net_top_move), learning_rate)

  epoch = 0
  while epoch < numOfEpochs
    epoch += 1
    data = collectData(net_top_move, net_bot_move, lengthOfBuffer, r_end, discount, length_of_game_tolerance)

    for i in 1:lengthOfBuffer
      (data[2][i] < 5) && (Flux.train!(loss_move, zip(transformState(data[1][:,i]),data[2][i],data[3][i]), opt_move))
    end

    if (epoch % 100 == 0)
      println("$epoch")
      known_state = [1; 2; 5; 2; 1]
      n0 = net_top_move(transformState(known_state)).data
      println("Safety check moving from start: $(n0)")
    end

    #check against random net
    if (epoch % 1000 == 0 || epoch == 1)
      dummy_games = [game(net_top_move, net_bot_move) for i in 1:1000]
      net_top_wins = sum(x[1] == :top_player_won for x in dummy_games)
      avg_length = mean(x[2] for x in dummy_games)
      println("Epoch: $(epoch), net_top won $net_top_wins against random net in $avg_length rounds")
    end
  end
end


train_hupo!()
game_show(net_top_move, net_bot_move)
