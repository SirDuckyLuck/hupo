#first six are coordinates for top player (row and column), in case of being out of the game it is 0 0
#second six are coordinates for bottom player
#next three are states for top player: 2 == has tottem, 1 == had totem, 0 == did not have totem, -1 == out of the game
#last three are states for bottom player
#actions are coded with two numbers: first is 1 for moving top, 2 for right, 3 for bottom, 4 for left.
#second number is giving stone to whom to give totem (1-3 top player, 4-6 bottom player)

function fill_state_beginning!(state)
    state[:] .= [1; 1; 1; 2; 1; 3; 5; 1; 5; 2; 5; 3; 0; 2; 0; 0; 0; 0]
end

function print_state(state::Array{Int})
    M = Array{String}(5,3)
    fill!(M,"-")
    M[3,2] = "x"
    for i in 1:6
        id = i*2-1
        state[id]==0 && continue
        M[state[id],state[id+1]] = string(state[12+i])
    end
    Base.showarray(STDOUT,M,false)
end

function action(state, net, exploration=0.)
    a = Array{Bool}(4,6) #which direction, #whom to pass
    fill!(a,true)
    active = find(state[13:end] .== 2)
    stone_position = [state[active*2-1];state[active*2]]

    for move in 1:4 #check moves
        new_position = copy(stone_position)
        if move == 1
            new_position[1] -=1
        elseif move == 2
            new_position[2] +=1
        elseif move == 3
            new_position[1] +=1
        elseif move == 4
            new_position[2] -=1
        end

        if new_position == [3; 2]
            a[move,:] .= false
        end
        for stone in 1:6
            if new_position == state[stone*2-1:stone*2]
                a[move,:] .= false
            end
        end
    end

    for pass in 1:6
        if state[12+pass] !=0
            a[:,pass] .= false
        end
    end

    v = rand()<exploration ? rand(4*6):rand(4*6)#net(state)

    v[.!vec(a)] = 0.0
    v ./= sum(v)
    value, best_move = findmax(v)
    if isnan(value)
        println("cant move")
        return "error"
    end

    move_to = mod(best_move,4)==0 ? 4:mod(best_move,4)
    pass_to = mod(best_move,4)==0 ? div(best_move,4):div(best_move,4)+1
    move_to, pass_to
end

function execute!(state,a)
    won = ""
    reward = 0.
    active = find(state[13:end] .== 2)[1]

    #move the stone
    if a[1] == 1
        state[active*2-1] -=1
    elseif a[1] == 2
        state[active*2] +=1
    elseif a[1] == 3
        state[active*2-1] +=1
    elseif a[1] == 4
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
    if  state[active*2-1] ∈ [0; 6] || state[active*2] ∈ [0 4]
        state[(active*2-1):(active*2)] = [0;0]
        state[12+active] = -1
        reward = -1.
    end

    #check if anyone won
    if state[1:2] == [4;2] || state[3:4] == [4;2] || state[5:6] == [4;2] || state[16:18] == [-1; -1; -1]
        won = "top player won"
    elseif state[7:8] == [2;2] || state[9:10] == [2;2] || state[11:12] == [2;2] || state[13:15] == [-1; -1; -1]
        won = "bottom player won"
    end

    won, reward
end

function game(net_top, net_bot, exploration)
    states_top = Array{Int}(0,18)
    rewards_top = Vector{Float64}()
    states_bot = Array{Int}(0,18)
    rewards_bot = Vector{Float64}()

    state = Array{Int}(6*2+6)
    fill_state_beginning!(state)
    active_player = "top"

    while true
        a = active_player == "top" ? action(state, net_top, exploration):action(state, net_bot, exploration)
        won, reward = execute!(state,a)
        if active_player=="top"
            states_top = vcat(states_top,deepcopy(state)')
            push!(rewards_top, reward)
            (won=="top player won") && (rewards_top[end]+=3)
            (won=="bottom player won") && (rewards_top[end]-=3)
        else
            states_bot = vcat(states_bot,deepcopy(state)')
            push!(rewards_bot, reward)
            (won=="top player won") && (rewards_bot[end]-=3)
            (won=="bottom player won") && (rewards_bot[end]+=3)
        end

        if !(won=="")
            break
        end
        active_player = active_player == "top" ? "bottom":"top"
    end

    states_top, rewards_top, states_bot, rewards_bot
end


function game_show(net_top, net_bot)
    state = Array{Int}(6*2+6)
    fill_state_beginning!(state)
    active_player = "top"
    round_number = 1

    while true
        println("Round $(round_number)")
        print_state(state)
        println()
        a = active_player == "top" ? action(state, net_top):action(state, net_bot)
        won, _ = execute!(state,a)
        if !(won=="")
            println(won)
            break
        end
        active_player = active_player == "top" ? "bottom":"top"
        round_number += 1
    end
end

st, rt, sb, rb = game([],[],0.1)
game_show(5,7)
