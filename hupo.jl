include("UnicodeGrids.jl")
include("printing.jl")
include("collectingData.jl")
using .UnicodeGrids
using Statistics
using Flux

@enum Move up = 1 right = 2 down = 3 left = 4 out = 5


function get_player_with_token(state)
  findfirst(state[13:end], 2)
end


function fill_state_beginning!(state)
  state[:] .= [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
end


function sample_move(state, active_stone, net_move)
  p = net_move(state).data .+ 1e-6

  stone_position = [state[active_stone*2 - 1];state[active_stone*2]]

  for move in instances(Move)[1:4] # check moves plausibility except getting out
    new_position = copy(stone_position)
    if move == up
        new_position[1] -=1
    elseif move == right
        new_position[2] +=1
    elseif move == down
        new_position[1] +=1
    elseif move == left
        new_position[2] -=1
    end

    # check for middle
    if new_position == [3; 2]
        p[Int(move)] = 0.
    end
    # check if there is another stone
    for stone in 1:6
        if new_position == state[stone*2-1:stone*2]
            p[Int(move)] = 0.
        end
    end
    # check if out of the board
    if new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
        p[Int(move)] = 0.
    end
  end

  (sum(p[1:4]) > 0.) && (p[5] = 0.) # if you can do something else than get kicked, do
  p ./= sum(p)
  r = rand()
  move = Move(findfirst(x -> x >= r, cumsum(p)))
end


function sample_pass(state, active_stone, net_pass)
  p = net_pass(state).data .+ 1e-6

  for pass in 1:6 # check passing plausibility
      if state[12+pass] != 0.
          p[pass] = 0.
      end
  end

  p ./= sum(p)

  r = rand()
  pass = findfirst(x -> x >= r, cumsum(p))
end


function apply_move!(state, active_stone, move)
  if move == up
      state[active_stone*2 - 1] -=1
  elseif move == right
      state[active_stone*2] +=1
  elseif move == down
      state[active_stone*2 - 1] +=1
  elseif move == left
      state[active_stone*2] -=1
  elseif move == out
        state[(active_stone*2-1):(active_stone*2)] = [0;0]
        state[12+active_stone] = -1
  end

  active_stone
end

function apply_pass!(state, active_stone, pass)
  state[12 + pass] = 2
  if active_stone ∈ [1 2 3] && pass ∈ [1 2 3]
      state[12 + active_stone] = 1
  elseif active_stone ∈ [1 2 3] && pass ∈ [4 5 6]
      state[12 + active_stone] = 0
      for s in 13:18
          if state[s] == 1
              state[s] = 0
          end
      end
  elseif active_stone ∈ [4 5 6] && pass ∈ [4 5 6]
      state[12 + active_stone] = 1
  elseif active_stone ∈ [4 5 6] && pass ∈ [1 2 3]
      state[12 + active_stone] = 0
      for s in 13:18
          if state[s] == 1
              state[s] = 0
          end
      end
  end

  pass
end


function check_state(state, active_stone)
  won = Symbol()

  if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
      won = :top_player_won
  elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
      won = :bottom_player_won
  end

  if active_stone ∈ [1;2;3]
    active_player = :top
  else
    active_player = :bot
  end

  won, active_player
end


function game(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = :top
    active_stone = 2

    while true
      move = active_player == :top ? sample_move(state, active_stone, net_top_move) : sample_move(state, active_stone, net_bot_move)
      active_stone = apply_move!(state, active_stone, move)
      pass = active_player == :top ? sample_pass(state, active_stone, net_top_pass) : sample_pass(state, active_stone, net_bot_pass)
      active_stone = apply_pass!(state, active_stone, pass)
      won, active_player = check_state(state, active_stone)

      if won ∈ [:top_player_won :bottom_player_won]
          return won
      end
    end
end


# game(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
