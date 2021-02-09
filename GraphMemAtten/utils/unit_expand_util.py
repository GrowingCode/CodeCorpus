from meta_info.non_hyper_constant import skeleton_e, skeleton_pe, skeleton_one,\
  all_skt_pe_to_each_base, all_skt_pe_to_each_start, all_skt_pe_to_each_end,\
  all_skt_one_to_each_base, all_skt_one_to_each_start, all_skt_one_to_each_end
from meta_info.hyper_parameter import skeleton_mode, n_skt


def get_unit_expand_sequence(ens, ens_len, replace_unk_with_none=True):
  need_expand = 0
  if skeleton_mode == skeleton_e:
    pass
  elif skeleton_mode == skeleton_pe:
    need_expand = 1
    unit_expand_base = all_skt_pe_to_each_base
    unit_expand_start = all_skt_pe_to_each_start
    unit_expand_end = all_skt_pe_to_each_end
  elif skeleton_mode == skeleton_one:
    need_expand = 1
    unit_expand_base = all_skt_one_to_each_base
    unit_expand_start = all_skt_one_to_each_start
    unit_expand_end = all_skt_one_to_each_end
  else:
    assert False
  
  seq = []
#   print("length of ens:" + str(len(ens)))
  for en in ens:
    if 2 < en < n_skt and need_expand:
      en_start = unit_expand_start[en]
      en_end = unit_expand_end[en]
      assert en_end >= en_start, "wrong en:" + str(en)
      tens = unit_expand_base[en_start:en_end+1].tolist()
    else:
      tens = [en]
    for ten in tens:
      if replace_unk_with_none:
        if 0 <= ten <= 2 or n_skt <= ten <= n_skt + 2:
          seq.append(None)
        else:
          seq.append(ten)
      else:
        seq.append(ten)
      
  if ens_len >= 0:
    seq = seq[0:ens_len]
  return seq


def get_unit_expand_sequence_list(ens_list, ens_len):
  res = []
  for ens in ens_list:
    r_ens = get_unit_expand_sequence(ens, ens_len, replace_unk_with_none=False)
    res.append(r_ens)
  return res
    



