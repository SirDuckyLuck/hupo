@enum Move up = 1 right = 2 down = 3 left = 4
const d_moves = Dict(up => "↑", right => "→", down => "↓", left => "←")

function game_show(net_top_move, net_bot_move)
    state = Array{Int}(undef,5)
    fill_state_beginning!(state)
    active_player = :top
    round_number = 1

    println()
    println("Round $(round_number)")
    print_state(state)
    println("press <Enter>")

    while true
        readline()
        clear(15)

        idx = state[5]

        move = active_player == :top ? sample_move(state, net_top_move) : sample_move(state, net_bot_move)
        apply_move!(state, move)
        won = check_state(state)

        println("Round $(round_number)")
        print_state(state)
        println("player $idx moves $(d_moves[move])  and passes token to player $(state[5])")

        if won ∈ (:top_player_won, :bottom_player_won)
            println(won)
            break
        end
        round_number += 1
    end
end


function print_state(state::Array{Int})
    M = fill(" ", 5, 3)
    M[3,2] = "x"
    for i in 1:2
        c = string(i)
        state[5] == i && (c = aesRed * c * aesClear)
        id = i*2-1
        M[state[id],state[id+1]] = c
    end
    print_grid(M)
end


"Clear `n` lines above cursor."
function clear(n::Int)
    println("\033[$(n)A" * "\033[K\n" ^ n * "\033[$(n)A")
end
