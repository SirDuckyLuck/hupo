module UnicodeGrids

export grid

#         0 	1 	2 	3 	4 	5 	6 	7 	8 	9 	A 	B 	C 	D 	E 	F
#U+250x 	─ 	━ 	│ 	┃ 	┄ 	┅ 	┆ 	┇ 	┈ 	┉ 	┊ 	┋ 	┌ 	┍ 	┎ 	┏
#U+251x 	┐ 	┑ 	┒ 	┓ 	└ 	┕ 	┖ 	┗ 	┘ 	┙ 	┚ 	┛ 	├ 	┝ 	┞ 	┟
#U+252x 	┠ 	┡ 	┢ 	┣ 	┤ 	┥ 	┦ 	┧ 	┨ 	┩ 	┪ 	┫ 	┬ 	┭ 	┮ 	┯
#U+253x 	┰ 	┱ 	┲ 	┳ 	┴ 	┵ 	┶ 	┷ 	┸ 	┹ 	┺ 	┻ 	┼ 	┽ 	┾ 	┿
#U+254x 	╀ 	╁ 	╂ 	╃ 	╄ 	╅ 	╆ 	╇ 	╈ 	╉ 	╊ 	╋ 	╌ 	╍ 	╎ 	╏
#U+255x 	═ 	║ 	╒ 	╓ 	╔ 	╕ 	╖ 	╗ 	╘ 	╙ 	╚ 	╛ 	╜ 	╝ 	╞ 	╟
#U+256x 	╠ 	╡ 	╢ 	╣ 	╤ 	╥ 	╦ 	╧ 	╨ 	╩ 	╪ 	╫ 	╬ 	╭ 	╮ 	╯
#U+257x 	╰ 	╱ 	╲ 	╳ 	╴ 	╵ 	╶ 	╷ 	╸ 	╹ 	╺ 	╻ 	╼ 	╽ 	╾ 	╿

function grid(matrix::Matrix{<:AbstractString}, squares::NTuple{2, Integer}...)
  isempty(matrix) && return ""

  m, n = size(matrix)
  A = Matrix{String}(2m+1, 2n+1)
  widths = [2 + maximum(strwidth, view(matrix, :, i)) for i ∈ 1:n]'

  # corners
  A[1, 1] = "┌"
  A[1, end] = "┐"
  A[end, 1] = "└"
  A[end, end] = "┘"

  # edges
  A[1, 3:2:end-2] .= "┬"
  A[end, 3:2:end-2] .= "┴"
  A[3:2:end-2, 1] .= "├"
  A[3:2:end-2, end] .= "┤"

  # middle
  A[3:2:end-2, 3:2:end-2] .= "┼"
  A[1:2:end, 2:2:end] .= "─" .^ widths
  A[2:2:end, 1:2:end] .= "│"
  A[2:2:end, 2:2:end] .= ((s, w) -> rpad(" $s ", w)).(matrix, widths)

  # highlight squares
  s = Set(squares)
  for (x, y) ∈ squares
    if x ∈ 1:m && y ∈ 1:n
      A[2x, [2y-1, 2y+1]] .= "┃"
      A[[2x-1, 2x+1], 2y] .= "━" ^ widths[y]
      highlight_corner!(A, s, x, y, -1, -1, "┏","┳","┲","┣","┢","╋","╊","╈","╆")
      highlight_corner!(A, s, x, y, -1, +1, "┓","┳","┱","┫","┪","╋","╉","╈","╅")
      highlight_corner!(A, s, x, y, +1, -1, "┗","┻","┺","┣","┡","╋","╊","╇","╄")
      highlight_corner!(A, s, x, y, +1, +1, "┛","┻","┹","┫","┩","╋","╉","╇","╃")
    end
  end

  join((join(A[i, :]) for i ∈ 1:size(A, 1)), "\n")
end


function highlight_corner!(A, s, x, y, dx, dy, c1,c21,c22,c31,c32,c41,c42,c43,c44)
  ax, ay = 2x + dx, 2y + dy
  a = dx < 0 ? 1 : size(A, 1)
  b = dy < 0 ? 1 : size(A, 2)
  if (ax, ay) == (a, b)
    A[ax, ay] = c1
  elseif ax == a
    A[ax, ay] = (x, y+dy) ∈ s ? c21 : c22
  elseif ay == b
    A[ax, ay] = (x+dx, y) ∈ s ? c31 : c32
  else
    if (x+dx, y+dy) ∈ s || ((x+dx, y), (x, y+dy)) ⊆ s
      A[ax, ay] = c41
    elseif (x+dx, y) ∈ s
      A[ax, ay] = c42
    elseif (x, y+dy) ∈ s
      A[ax, ay] = c43
    else
      A[ax, ay] = c44
    end
  end
end

end # module


# using .UnicodeGrids
#
# m = ["o" " " " "; "x" "o" ""; "x" "x" "o"]
# println(grid(m, [(x, x) for x = 1:3]...))
#
# m = permutedims(reshape(split("""
# What if God was one of us?
# Just a slob like one of us
# Just a stranger on the bus
# Tryin' to make his way home?""")[1:21], 7, 3), (2, 1))
# println(grid(m))
#
# m = fill(" ", 8, 8)
# m[4, 5] = "♞"
# println(grid(m, (2,4), (2,6), (3,3), (3,7), (5,3), (5,7), (6,4), (6,6)))
