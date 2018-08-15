include("hupo.jl")

net_top = Chain(
  Dense(18, 100, relu),
  Dense(100, 100, relu),
  Dense(100, 30),
  softmax)

net_bot = Chain(
  Dense(18, 30),
  softmax)

function loss(states,actions,rewards)
    p_all = net_top(states)
    p = sum(Flux.onehotbatch(actions,collect(1:30)) .* p_all, 1)
    Flux.crossentropy(p, rewards)
end

global numOfEpochs = 1001
global lengthOfBuffer = 400
global r_end = 1.
global r_add = 5.
global discount = 0.75



function train_hupo!(net_top)
  opt = ADAM(Flux.params(net_top))
  net_bot = Chain(
    Dense(18, 30),
    softmax)

  for epoch in 1:numOfEpochs
    mb = memory_buffer(lengthOfBuffer)
    collectData!(net_top, net_top, mb, r_end, r_add, discount)

    if !isnan(std(mb.rewards))
      mb.rewards .= (mb.rewards .- mean(mb.rewards))./std(mb.rewards)
    end

    data = Iterators.repeated((mb.states, mb.actions, mb.rewards), 1)

    opt = ADAM(Flux.params(net_top))
    Flux.train!(loss, data, opt)

    # survival of the fittest
    if (epoch % 100 == 0)
      candidate = deepcopy(net_top)
      deciding_games = [game(candidate, net_top) for i in 1:100]
      candidate_wins = sum(deciding_games.=="top player won")
      println("Epoch: $(epoch), candidate won $candidate_wins")
      if candidate_wins > 50
        net_top = deepcopy(candidate)
      end

      # safety check --- the probability of going right should rise
      known_state = [1; 1; 4; 1; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
      println(sum(net_top(known_state).data[2:5:30]))
    end
    #check against random net
    if (epoch % 1001 == 0)
      dummy_games = [game(net_top, net_bot) for i in 1:1000]
      net_top_wins = sum(dummy_games.=="top player won")
      println("Epoch: $(epoch), net_top won $net_top_wins against random net")
    end
  end
end

train_hupo!(net_top)
# game_show(net_top, net_bot)
