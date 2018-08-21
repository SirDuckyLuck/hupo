for i in 1:length(params(net_top_move))
  params(net_top_move)[i].data .= 0
end
for i in 1:length(params(net_top_pass))
  params(net_top_pass)[i].data .= 0
end
