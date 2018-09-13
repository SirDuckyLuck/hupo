include("hupo.jl")
using BSON: @save
using BSON: @load


numOfEpochs = 10 ^ 5 + 1
const lengthOfBuffer = 300
const r_end = 1.
const discount = 0.99
const learning_rate = 1e-4
const length_of_game_tolerance = 200

G = [Chain(
    Dense(72, 100, relu),
    Dense(100, 100, relu),
    Dense(100, 30),
    softmax)]


function loss(states::Matrix{Int}, actions, rewards)
  Flux.crossentropy(candidate(states)[Flux.onehotbatch(actions, 1:30)] .+ 1e-8, rewards)
end


function loss(state::Vector{Int}, action, reward)
  p = candidate(state)[action]
  -log(p + 1e-8) * reward
end


function train_hupo!(;n_epochs = numOfEpochs, level = :original)
  println("training $n_epochs epochs, level = $level")
  candidate = deepcopy(G[1])
  opt = SGD(Flux.params(candidate), learning_rate)
  # opt = ADAM(Flux.params(net_top))

  epoch = 0
  while epoch < n_epochs
    #check against random net
    if epoch % 1000 == 0
      println("Epoch: $(epoch)")
      wins = test_candidate(candidate, G)
      if wins >= 550
        push!(G, candidate)
        candidate = deepcopy(G[end])
        opt = SGD(Flux.params(candidate), learning_rate)
      end
    end

    epoch += 1
    data = collectData(candidate, G[rand(1:length(G))], lengthOfBuffer, r_end, discount, length_of_game_tolerance, level)

    # Flux.train!(loss, [(data[1][:,i],data[2][i],data[3][i]) for i in 1:lengthOfBuffer], opt)
    Flux.train!(loss, [(data...)], opt)

    if (epoch % 100 == 0)
      println("$epoch")
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


function test_models(net_top, net_bot)
  test_games = [game(net_player(net_top), net_player(net_bot)) for i in 1:1000]
  net_top_wins = sum(x[1] == :top_player_won for x in test_games)
  avg_length = mean(x[2] for x in test_games)
  println("top net won $net_top_wins against bottom net in $avg_length rounds")
end


function test_candidate(candidate, G)
  test_games = [game(net_player(candidate), net_player(G[rand(1:length(G))])) for i in 1:1000]
  candidate_wins = sum(x[1] == :top_player_won for x in test_games)
  avg_length = mean(x[2] for x in test_games)
  println("candidate won $candidate_wins against previous generations in $avg_length rounds")

  candidate_wins
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

println("Use one of these: train_hupo!(), @save_models, load_models!(), @test_models, @play")
