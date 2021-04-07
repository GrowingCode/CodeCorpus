import pprint


def pretty_print_dict(dd):
  ll_list = sorted(dd.items(), key = lambda d:d[1])
  
  pp = pprint.PrettyPrinter(indent=2)
  pp.pprint(ll_list)

#   
#   table = PrettyTable(["key","value"])
#   table.align["key"] = "l"
#   table.padding_width = 1
#   for tup in list:
#     table.add_row([tup[0], tup[1]])
#    
#   print(table)



