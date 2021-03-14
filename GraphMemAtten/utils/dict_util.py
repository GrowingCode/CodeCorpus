


def gen_dict_with_str_key_to_num_key(dc):
  res = {}
  for key in dc:
#     print(key+':'+dc[key])
    res[int(key)] = dc[key]
  return res





