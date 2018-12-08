mutable struct memory_buffer
  N::Int
  k::Int
  k_init::Int
  states::Matrix{Int}
  actions::Vector{Int}
  rewards::Vector{Float64}
end


function action2idx(move::Move, pass::Int)
  (Int(move) - 1)*6 + pass
end


function memory_buffer(N::Int)
  memory_buffer(N, 1, 1, zeros(Int, 72, N), zeros(N), zeros(N))
end


function game!(net_top, net_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)
  state = state_beginning(level)
  active_player = :top
  won = Symbol()
  game_length = 0
  mb_top.k_init = mb_top.k
  mb_bot.k_init = mb_bot.k

  while true
    game_length += 1
    move, pass = active_player == :top ?
                 sample_action(state, net_top) :
                 sample_action(state, net_bot)
    if (active_player == :top)
      mb = mb_top
    else
      mb = mb_bot
    end

    if mb.k <= mb.N
      mb.states[:,mb.k] = transformState(state)
      mb.actions[mb.k] = action2idx(move, pass)
      mb.k += 1
    end

    active_stone = apply_move!(state, move)
    apply_pass!(state, active_stone, pass)
    won = check_state(state)
    active_player = get_active_player(state)

    if game_length > length_of_game_tolerance
      won = :bottom_player_won
    end

    if won ∈ (:top_player_won, :bottom_player_won)
      mb = mb_top
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - mb.k_init - 1):-1:0)) .* r_end
        if won == :top_player_won
            mb.rewards[mb.k_init:(mb.k - 1)] .+= reward
        else
            mb.rewards[mb.k_init:(mb.k - 1)] .-= reward
        end
      end
      mb = mb_bot
      if mb.k <= mb.N
        reward = (discount .^ ((mb.k - mb.k_init - 1):-1:0)) .* r_end
        if won == :bottom_player_won
            mb.rewards[mb.k_init:(mb.k - 1)] .+= reward
        else
            mb.rewards[mb.k_init:(mb.k - 1)] .-= reward
        end
      end
      return won, game_length
    end
  end
end


function collectData(net_top, net_bot, lengthOfBuffer, r_end, discount, length_of_game_tolerance, level)
  mb_top = memory_buffer(lengthOfBuffer)
  mb_bot = memory_buffer(lengthOfBuffer)

  net_top_wins = 0
  n_games = 0
  sum_game_lenghts = 0

  while (mb_top.k <= mb_top.N) || (mb_bot.k <= mb_bot.N)
    won, game_length = game!(net_top, net_bot, mb_top, mb_bot, r_end, discount, length_of_game_tolerance, level)

    net_top_wins += won == :top_player_won ? 1 : 0
    n_games += 1
    sum_game_lenghts += game_length
  end

  global mb_top = mb_top
  global mb_bot = mb_bot

  mb_top.rewards = (mb_top.rewards .- mean(mb_top.rewards))./std(mb_top.rewards)
  mb_bot.rewards = (mb_bot.rewards .- mean(mb_bot.rewards))./std(mb_bot.rewards)
  data = (mb_top, mb_bot, (n_games, net_top_wins, sum_game_lenghts))
end


# data = collectData(net_top, net_bot)
