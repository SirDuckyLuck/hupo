include("UnicodeGrids.jl")
include("printing.jl")
include("collectingData.jl")
using .UnicodeGrids
using Statistics
using Flux

@enum Move up right down left out


function get_player_with_token(state)
    findfirst(state[13:end], 2)
end


function fill_state_beginning!(state)
    state[:] .= [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
end


function action(state, net_move, net_pass)
  active = findall(state[13:end] .== 2)
  stone_position = [state[active*2 .- 1];state[active*2]]

  p = net_move(state).data .+ 1e-6

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
          p[Int(move)+1] = 0.
      end
      # check if there is another stone
      for stone in 1:6
          if new_position == state[stone*2-1:stone*2]
              p[Int(move)+1] = 0.
          end
      end
      # check if out of the board
      if new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
          p[Int(move)+1] = 0.
      end
  end

  (sum(p[1:4]) > 0.) && (p[5] = 0.) # if you can do something else than get kicked, do
  p ./= sum(p)
  r = rand()
  move = Move(findfirst(x -> x >= r, cumsum(p)) - 1)
  ################

  p = net_pass(state).data .+ 1e-6

  for pass in 1:6 # check passing plausibility
      if state[12+pass] != 0.
          p[pass] = 0.
      end
  end

  p ./= sum(p)

  r = rand()
  pass = findfirst(x -> x >= r, cumsum(p))

  move, pass
end


function execute!(state, a)
    won = Symbol()
    active = findall(state[13:end] .== 2)[1]

    #move the stone
    if a[1] == up
        state[active*2-1] -=1
    elseif a[1] == right
        state[active*2] +=1
    elseif a[1] == down
        state[active*2-1] +=1
    elseif a[1] == left
        state[active*2] -=1
    end

    #hand the totem
    state[12+a[2]] = 2
    if active ∈ [1 2 3] && a[2] ∈ [1 2 3]
        state[12+active] = 1
    elseif active ∈ [1 2 3] && a[2] ∈ [4 5 6]
        state[12+active] = 0
        for s in 13:18
            if state[s] == 1
                state[s] = 0
            end
        end
    elseif active ∈ [4 5 6] && a[2] ∈ [4 5 6]
        state[12+active] = 1
    elseif active ∈ [4 5 6] && a[2] ∈ [1 2 3]
        state[12+active] = 0
        for s in 13:18
            if state[s] == 1
                state[s] = 0
            end
        end
    end

    #check if stone is out of the game
    if  state[active*2-1] ∈ [0; 6] || state[active*2] ∈ [0 4] || a[1] == out
        state[(active*2-1):(active*2)] = [0;0]
        state[12+active] = -1
        (active == 2) && (won = :bot_player_lost_a_stone)
    end

    #check if anyone won
    if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
        won = :top_player_won
    elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
        won = :bottom_player_won
    end

    if findall(state[13:end] .== 2)[1] ∈ [1;2;3]
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

    while true
        a = active_player == :top ? action(state, net_top_move, net_top_pass) : action(state, net_bot_move, net_bot_pass)

        won, active_player = execute!(state,a)

        if won ∈ [:top_player_won :bottom_player_won]
            return won
        end
    end
end
