include("UnicodeGrids.jl")
using .UnicodeGrids
using Statistics

@enum Move up right down left out

const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←", out => "~")

mutable struct memory_buffer
  N::Int
  states::Array{Int}
  actions::Vector{Float64}
  rewards::Vector{Float64}
end

function memory_buffer(N::Int)
  memory_buffer(N, Array{Int}(undef,18,N), zeros(N), zeros(N))
end

"Clear `n` lines above cursor."
function clear(n::Int)
    println("\033[$(n)A" * "\033[K\n" ^ n * "\033[$(n)A")
end

function get_player_with_token(state)
    findfirst(state[13:end], 2)
end

function fill_state_beginning!(state)
    state[:] .= [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
end

function print_state(state::Array{Int})
    M = fill(" ", 5, 3)
    M[3,2] = "x"
    for i in 1:6
        state[12+i] == -1 && continue
        c = string(i)
        state[12+i] == 1 && (c = aesRed * c * aesClear)
        state[12+i] == 2 && (c = aesBold * aesYellow * c * aesClear)
        id = i*2-1
        M[state[id],state[id+1]] = c
    end
    print_grid(M)
end

function print_state(state::Array{Int}, pos, arrow)
    M = fill(" ", 5, 3)
    M[3,2] = "x"
    M[pos[1], pos[2]] = arrow
    for i in 1:6
        state[12+i] == -1 && continue
        c = string(i)
        state[12+i] == 1 && (c = aesRed * c * aesClear)
        state[12+i] == 2 && (c = aesBold * aesYellow * c * aesClear)
        id = i*2-1
        M[state[id],state[id+1]] = c
    end
    print_grid(M)
end

function action(state, net)
    a = Array{Bool}(undef,5,6) # which direction, #whom to pass
    fill!(a,true)
    active = findall(state[13:end] .== 2)
    stone_position = [state[active*2 .- 1];state[active*2]]

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
            a[Int(move)+1,:] .= false
        end
        # check if there is another stone
        for stone in 1:6
            if new_position == state[stone*2-1:stone*2]
                a[Int(move)+1,:] .= false
            end
        end
        # check if out of the board
        if new_position[1] ∈ [0; 6] || new_position[2] ∈ [0; 4]
            a[Int(move)+1,:] .= false
        end
    end

    for pass in 1:6 # check passing plausibility
        if state[12+pass] !=0
            a[:,pass] .= false
        end
    end

    # if there is a move other than drop out of the game and pass
    (sum(a[1:4,:]) > 0) && (a[5,:] .= false)

    v = net(state).data
    v[.!vec(a)] .= 0.0
    v ./= sum(v)

    r = rand()
    best_move = findfirst(x -> x > r, cumsum(v))

    if best_move ∈ 1:5:30
      move_to = up
    elseif best_move ∈ 2:5:30
      move_to = right
    elseif best_move ∈ 3:5:30
      move_to = down
    elseif best_move ∈ 4:5:30
      move_to = left
    else
      move_to = out
    end

    if best_move ∈ 1:5
      pass_to = 1
    elseif best_move ∈ 6:10
      pass_to = 2
    elseif best_move ∈ 11:15
      pass_to = 3
    elseif best_move ∈ 16:20
      pass_to = 4
    elseif best_move ∈ 21:25
      pass_to = 5
    else
      pass_to = 6
    end

    (move_to, pass_to), best_move
end

function execute!(state, a)
    won = ""
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
        (active == 2) && (won = "bot player lost a stone")
    end

    #check if anyone won
    if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
        won = "top player won"
    elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
        won = "bottom player won"
    end

    won
end

function game!(net_top, net_bot, memory_buffer, k)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = "top"
    k_init = copy(k)

    while true
        a, a_idx = active_player == "top" ? action(state, net_top) : action(state, net_bot)
        if (active_player=="top") && (k <= memory_buffer.N)# collect data for top player
            memory_buffer.states[:,k] = state
            memory_buffer.actions[k] = a_idx
            k += 1
        end

        won = execute!(state,a)

        if won ∈ ["top player won" "bot player lost a stone"]
            memory_buffer.rewards[k-1] += 5.
        end

        if won ∈ ["top player won" "bottom player won"]
            if won=="top player won"
                memory_buffer.rewards[k_init:min(k - 1,memory_buffer.N)] .+= 1
            else
                memory_buffer.rewards[k_init:min(k - 1,memory_buffer.N)] .-= 1
            end
            return k
        end

        active_player = active_player == "top" ? "bottom" : "top"
    end
end


function game_show(net_top, net_bot)
    state = Array{Int}(undef,6*2+6)
    fill_state_beginning!(state)
    active_player = "top"
    round_number = 1

    println()
    println("Round $(round_number)")
    print_state(state)
    println("press <Enter>")

    while true
        readline()
        clear(15)

        idx = get_player_with_token(state)
        pos = (state[2*idx - 1], state[2*idx])
        a, p = active_player == "top" ? action(state, net_top) : action(state, net_bot)
        won = execute!(state, a)

        println("Round $(round_number)")
        print_state(state, pos, d_moves[a[1]])
        println("player $idx moves $(d_moves[a[1]])  and passes token to player $(a[2])")

        if !(won=="")
            println(won)
            break
        end
        active_player = active_player == "top" ? "bottom" : "top"
        round_number += 1
    end
end
