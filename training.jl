include("hupo.jl")

numOfEpochs = 10 ^ 3
const lengthOfBuffer = 300
const r_end = 1.
const discount = 0.99
const learning_rate = 1e-4
const length_of_game_tolerance = 1001

const net_top_move = Chain(
    Dense(72, 32, relu),
    Dense(32, 4),
    softmax)
const net_top_pass = Chain(
    Dense(72, 1, relu),
    Dense(1, 6),
    softmax)
const net_bot_move(x) = softmax(param(ones(4, 72)) * x)
const net_bot_pass(x) = softmax(param(ones(6, 72)) * x)


function loss_move(state, move, reward)
  p = net_top_move(state)[move]
  -log(p + 1e-8) * reward
end

function loss_pass(state, pass, reward)
  p = net_top_pass(state)[pass]
  -log(p + 1e-8) * reward
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
    #check against random net
    if epoch % 1000 == 0
      dummy_games = [game(net_top_move, net_top_pass, net_bot_move, net_bot_pass) for i in 1:1000]
      net_top_wins = sum(x[1] == :top_player_won for x in dummy_games)
      avg_length = mean(x[2] for x in dummy_games)
      println("Epoch: $(epoch), net_top won $net_top_wins against random net in $avg_length rounds")
    end

    epoch += 1
    data = collectData(net_top_move, net_top_pass, net_bot_move, net_bot_pass, lengthOfBuffer, r_end, discount, length_of_game_tolerance)

    Flux.train!(loss_move, [(transformState(data[1][:,i]),data[2][i],data[5][i]) for i in 1:lengthOfBuffer if data[2][i] < 5], opt_move)
    # Flux.train!(loss_pass, [(transformState(data[3][:,i]),data[4][i],data[5][i]) for i in 1:lengthOfBuffer], opt_pass)

    if (epoch % 100 == 0)
      println("$epoch")
      # known_state = [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      # n0 = net_top_move(transformState(known_state)).data
      # println("Safety check moving from start: $(n0)")
      # known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      # n1 = net_top_move(transformState(known_state)).data
      # println("Safety check moving: $(n1) which is $(n1[2]/(n1[1] + n1[2])*100) %")
      # known_state = [1; 1; 1; 2; 1; 3; 5; 1; 4; 1; 5; 2; 1; 2; 1; 0; 0; 0]
      # n2 = net_top_pass(transformState(known_state)).data
      # println("Safety check passing: $(n2) which is $(n2[4]/(n2[4] + n2[5] + n2[6])*100) %")
      # known_state = [1; 1; 4; 3; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      # n1 = net_top_move(transformState(known_state)).data
      # println("Safety check moving: $(n1) which is $(n1[4]/(n1[1] + n1[4])*100) %")
      # known_state = [1; 1; 5; 2; 1; 3; 4; 1; 4; 2; 4; 3; 1; 2; 1; 0; 0; 0]
      # n2 = net_top_pass(transformState(known_state)).data
      # println("Safety check passing: $(n2) which is $(n2[5]/(n2[4] + n2[5] + n2[6])*100) %")
      # known_state = [1; 1; 2; 1; 1; 2; 5; 1; 5; 2; 5; 3; 0; 0; 2; 0; 0; 0]
      # n2 = net_top_pass(transformState(known_state)).data
      # println("Safety check passing: $(n2) which is $((1-n2[1]/(n2[1] + n2[2] + n2[4] + n2[5] + n2[6]))*100) %")
      # println()
    end
  end
end


train_hupo!()
game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
using BSON: @save
using BSON: @load
@save joinpath(@__DIR__,"net_top_pass.bson") net_top_pass
@save joinpath(@__DIR__,"net_top_move.bson") net_top_move
@save joinpath(@__DIR__,"net_bot_pass.bson") net_bot_pass
@save joinpath(@__DIR__,"net_bot_move.bson") net_bot_move

function play_model()
  @load joinpath(@__DIR__,"net_top_pass.bson") net_top_pass
  @load joinpath(@__DIR__,"net_top_move.bson") net_top_move
  @load joinpath(@__DIR__,"net_bot_pass.bson") net_bot_pass
  @load joinpath(@__DIR__,"net_bot_move.bson") net_bot_move
  game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
end
