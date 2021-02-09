from meta_info.non_hyper_constant import skeleton_e, skeleton_pe, skeleton_one,\
  all_skt_pe_to_each_base, all_skt_pe_to_each_start, all_skt_pe_to_each_end,\
  all_skt_one_to_each_base, all_skt_one_to_each_start, all_skt_one_to_each_end
from meta_info.hyper_parameter import skeleton_mode


def get_unit_expand_sequence(ens, ens_len):
  if skeleton_mode == skeleton_e:
    seq = []
    seq.extend(ens)
    return seq
  elif skeleton_mode == skeleton_pe:
    unit_expand_base = all_skt_pe_to_each_base
    unit_expand_start = all_skt_pe_to_each_start
    unit_expand_end = all_skt_pe_to_each_end
  elif skeleton_mode == skeleton_one:
    unit_expand_base = all_skt_one_to_each_base
    unit_expand_start = all_skt_one_to_each_start
    unit_expand_end = all_skt_one_to_each_end
  else:
    assert False
  
  seq = []
  for en in ens:
    en_start = unit_expand_start[en]
    en_end = unit_expand_end[en]
    assert en_end >= en_start, "wrong en:" + str(en)
    seq.extend(unit_expand_base[en_start:en_end+1].tolist())
  if ens_len >= 0:
    seq = seq[0:ens_len]
  return seq


def get_unit_expand_sequence_list(ens_list, ens_len):
  res = []
  for ens in ens_list:
    r_ens = get_unit_expand_sequence(ens, ens_len)
    res.append(r_ens)
  return res
    



