global epoch = 0
include("hupo.jl")
using BSON: @save
using BSON: @load


numOfEpochs = 10 ^ 5 + 1
const lengthOfBuffer = 1000
const r_end = 1.
const discount = 0.99
const learning_rate = 1e-4
const length_of_game_tolerance = 200


function init_models()
  global net_top = Chain(
      Dense(72, 100, relu),
      Dense(100,30),
      softmax)
  # global net_top = Chain(
  #     Dense(72, 100, relu),
  #     Dense(100, 100, relu),
  #     Dense(100, 30),
  #     softmax)
  # net_bot(x) = softmax(param(ones(30, 72)) * x)
  global net_bot = Chain(
      Dense(72, 100),
      Dense(100,30),
      softmax)
end

function loss(states::Matrix{Int}, actions, rewards)
  Flux.crossentropy(net_top(states)[Flux.onehotbatch(actions, 1:30)] .+ 1e-8, rewards)
end

function loss_bot(states::Matrix{Int}, actions, rewards)
  Flux.crossentropy(net_bot(states)[Flux.onehotbatch(actions, 1:30)] .+ 1e-8, rewards)
end

function train_hupo!(;n_epochs = numOfEpochs, train_bot = false, level = :original)
  println("training $n_epochs epochs, level = $level")
  # opt = SGD(Flux.params(net_top), learning_rate)
  if train_bot
    opt = ADAM(Flux.params(net_bot))
  else
    opt = ADAM(Flux.params(net_top))
  end

  epoch = 0
  loss_list = Float64[]
  while epoch < n_epochs
    #check against random net
    if epoch % 100 == 0
      println("Epoch: $(epoch)")
      test_models(net_top, net_bot)

      if epoch > 1
        avg_loss = mean(loss_list)
        println("average loss: $avg_loss")
        empty!(loss_list)
      end

      params_net_top = collect(Iterators.flatten([copy(params(net_top)[i].data) for i in 1:length(params(net_top))]))
      println("net length: $(norm(params_net_top))")
      params_net_bot = collect(Iterators.flatten([copy(params(net_bot)[i].data) for i in 1:length(params(net_bot))]))
      println("net bot length: $(norm(params_net_bot))")

      if epoch>1
        net_mse = norm(params_net_top-params_net_top_before)
        println("net diff mse: $net_mse")
      end
      params_net_top_before = params_net_top

    end

    epoch += 1

    data_top, data_bot, (n_games, net_top_wins, sum_game_lenghts) = collectData(net_top, net_bot, lengthOfBuffer, r_end, discount, length_of_game_tolerance, level)

    global d_bot = data_bot
    global d_top = data_top

    if train_bot
      push!(loss_list, loss_bot(data_bot.states, data_bot.actions, data_bot.rewards).data)
    else
      push!(loss_list, loss(data_top.states, data_top.actions, data_top.rewards).data)
    end

    # Flux.train!(loss, [(data...) for i in 1:10], opt)
    if train_bot
      Flux.train!(loss_bot, [(data_bot.states, data_bot.actions, data_bot.rewards)], opt)
    else
      Flux.train!(loss, [(data_top.states, data_top.actions, data_top.rewards)], opt)
    end

    # params(net_top) .- params_net_top_before

    if (epoch % 100 == 0)
      # println("$epoch")
      # known_state = [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      # n0 = net_top_move(transformState(known_state)).data
    end
  end
end


function save_models(net_top, net_bot, name)
  @save joinpath(@__DIR__, "net_top$name.bson") net_top
  @save joinpath(@__DIR__, "net_bot$name.bson") net_bot
end

function load_models!(name = "")
  global net_top, net_bot
  @load joinpath(@__DIR__, "net_top$name.bson") net_top
  @load joinpath(@__DIR__, "net_bot$name.bson") net_bot
end

function load_2models!(name1 = "", name2="")
  global net_top, net_bot
  @load joinpath(@__DIR__, "net_top$name1.bson") net_top
  net_bot = net_top
  @load joinpath(@__DIR__, "net_top$name2.bson") net_top
end

function test_models(net_top, net_bot)
  dummy_games = [game(net_player(net_top), net_player(net_bot)) for i in 1:1000]
  net_top_wins = sum(x[1] == :top_player_won for x in dummy_games)
  avg_length = mean(x[2] for x in dummy_games)
  println("net_top won $net_top_wins against net_bot in $avg_length rounds")
end

function test_models_loss(net_top,net_bot)
  data_top, data_bot, (n_games, net_top_wins, sum_game_lenghts) = collectData(net_top, net_bot, 50*1000, r_end, discount, length_of_game_tolerance, :original)
  l_bot = loss_bot(data_bot.states, data_bot.actions, data_bot.rewards).data
  l_top = loss(data_top.states, data_top.actions, data_top.rewards).data

  l_top = round(l_top/ n_games,1)
  l_bot = round(l_bot/ n_games,1)
  net_top_win = round(net_top_wins / n_games * 100,1)
  avg_length = round(sum_game_lenghts / n_games,1)
  println("net_top won $net_top_win % with loss $l_top against net_bot with loss $l_bot in $avg_length rounds")
end

function test_models_swapped(net_top, net_bot)
  dummy_games = [game(net_player(net_bot), net_player(net_top)) for i in 1:1000]
  net_top_wins = sum(x[1] == :top_player_won for x in dummy_games)
  avg_length = mean(x[2] for x in dummy_games)
  println("net_bot won $net_top_wins against net_top in $avg_length rounds")
end

macro save_models(name = "")
  :(save_models(net_top, net_bot, $name))
end

macro test_models()
  :(test_models(net_top, net_bot))
end

macro play()
  :(game_show(net_top, net_bot))
end

println("Use one of these: init_models(), train_hupo!(), @save_models, load_models!(), @test_models, @play")
