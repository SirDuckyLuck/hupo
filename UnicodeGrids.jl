module UnicodeGrids

export print_grid
export aesRed
export aesGreen
export aesYellow
export aesBlue
export aesClear
export aesBold

#           0 	1 	2 	3 	4 	5 	6 	7 	8 	9 	A 	B 	C 	D 	E 	F
#U+250x 	─ 	━ 	│ 	┃ 	┄ 	┅ 	┆ 	┇ 	┈ 	┉ 	┊ 	┋ 	┌ 	┍ 	┎ 	┏
#
#U+251x 	┐ 	┑ 	┒ 	┓ 	└ 	┕ 	┖ 	┗ 	┘ 	┙ 	┚ 	┛ 	├ 	┝ 	┞ 	┟
#
#U+252x 	┠ 	┡ 	┢ 	┣ 	┤ 	┥ 	┦ 	┧ 	┨ 	┩ 	┪ 	┫ 	┬ 	┭ 	┮ 	┯
#
#U+253x 	┰ 	┱ 	┲ 	┳ 	┴ 	┵ 	┶ 	┷ 	┸ 	┹ 	┺ 	┻ 	┼ 	┽ 	┾ 	┿
#
#U+254x 	╀ 	╁ 	╂ 	╃ 	╄ 	╅ 	╆ 	╇ 	╈ 	╉ 	╊ 	╋ 	╌ 	╍ 	╎ 	╏
#
#U+255x 	═ 	║ 	╒ 	╓ 	╔ 	╕ 	╖ 	╗ 	╘ 	╙ 	╚ 	╛ 	╜ 	╝ 	╞ 	╟
#
#U+256x 	╠ 	╡ 	╢ 	╣ 	╤ 	╥ 	╦ 	╧ 	╨ 	╩ 	╪ 	╫ 	╬ 	╭ 	╮ 	╯
#
#U+257x 	╰ 	╱ 	╲ 	╳ 	╴ 	╵ 	╶ 	╷ 	╸ 	╹ 	╺ 	╻ 	╼ 	╽ 	╾ 	╿ 

const hl = "\u2500" # ─
const vl = "\u2502" # │
const tl = "\u250c" # ┌
const tr = "\u2510" # ┐
const bl = "\u2514" # └
const br = "\u2518" # ┘
const mm = "\u253c" # ┼
const mt = "\u252c" # ┬
const mb = "\u2534" # ┴
const ml = "\u251c" # ├
const mr = "\u2524" # ┤

# ANSI escape sequences
const aesRed = "\033[31m"
const aesGreen = "\033[32m"
const aesYellow = "\033[33m"
const aesBlue = "\033[34m"
const aesClear = "\033[0m"
const aesBold = "\033[1m"

function print_grid(M::Matrix{String})
    isempty(M) && return
    m, n = size(M)
    s  = tl * join(Iterators.repeated(hl^3, n), mt) * tr * '\n'
    for i ∈ 1:m - 1
        s *= vl*' ' * join(view(M, i, :), ' '*vl*' ') * ' '*vl * '\n'
        s *= ml * join(Iterators.repeated(hl^3, n), mm) * mr * '\n'
    end
    s *= vl*' ' * join(view(M, m, :), ' '*vl*' ') * ' '*vl * '\n'
    s *= bl * join(Iterators.repeated(hl^3, n), mb) * br * '\n'
    print(s)
end

end
