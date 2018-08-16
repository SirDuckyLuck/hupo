@enum Move up right down left out
const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←", out => "~")

function game_show(net_top_move, net_top_pass, net_bot_move, net_bot_pass)
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
        a = active_player == "top" ? action(state, net_top_move, net_top_pass) : action(state, net_bot_move, net_bot_pass)
        won, active_player = execute!(state, a)

        println("Round $(round_number)")
        print_state(state, pos, d_moves[a[1]])
        println("player $idx moves $(d_moves[a[1]])  and passes token to player $(a[2])")

        if won ∈ ["top player won" "bottom player won"]
            println(won)
            break
        end
        round_number += 1
    end
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


"Clear `n` lines above cursor."
function clear(n::Int)
    println("\033[$(n)A" * "\033[K\n" ^ n * "\033[$(n)A")
end
