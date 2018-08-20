include("hupo.jl")

numOfEpochs = 10 ^ 6
const lengthOfBuffer = 500
const r_end = 1.
const discount = 0.95
const learning_rate = 1e-5
const length_of_game_tolerance = 50
const minP = 1e-6
const maxP = 1-minP

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
const net_bot_move(x) = softmax(param(ones(5, 18)) * x)
const net_bot_pass(x) = softmax(param(ones(6, 18)) * x)
include("setParams.jl")


function loss_move(state, move, reward)
    p = net_top_move(state)[move]
    if minP < p < maxP
      -log(p) * reward
    else
      0p
    end
end

function loss_pass(state, pass, reward)
    p = net_top_pass(state)[pass]
    if minP < p < maxP
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
  opt_pass = SGD(Flux.params(net_top_pass), learning_rate)

  epoch = 0
  while epoch < numOfEpochs
    epoch += 1
    data = collectData(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, discount, length_of_game_tolerance)

    for i in 1:lengthOfBuffer
      Flux.train!(loss_move, zip(data[1][:,i],data[2][i],data[5][i]), opt_move)
      Flux.train!(loss_pass, zip(data[3][:,i],data[4][i],data[5][i]), opt_pass)
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
    if (epoch % 1000 == 0 || epoch == 1)
      dummy_games = [game(net_top_move, net_top_pass, net_bot_move, net_bot_pass) for i in 1:1000]
      net_top_wins = sum(dummy_games.==:top_player_won)
      println("Epoch: $(epoch), net_top won $net_top_wins against random net")
    end
  end
end


train_hupo!()
game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
