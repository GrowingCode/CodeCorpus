from meta_info.non_hyper_constant import skeleton_e, skeleton_pe, skeleton_one,\
  all_skt_pe_to_each_base, all_skt_pe_to_each_start, all_skt_pe_to_each_end,\
  all_skt_one_to_each_base, all_skt_one_to_each_start, all_skt_one_to_each_end,\
  np_int_type
from meta_info.hyper_parameter import skeleton_mode, n_skt


def get_unit_expand_sequence(ens, ens_len, infer_stage=False):
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
    if need_expand and ((not infer_stage) or (2 < en < n_skt)):
      en_start = unit_expand_start[en]
      en_end = unit_expand_end[en]
      assert en_end >= en_start, "wrong en:" + str(en)
      ll = unit_expand_base[en_start:en_end+1].tolist()
      for ll_o in ll:
        assert isinstance(ll_o, int)
      seq.append(ll)
    else:
      po_en = en
      if isinstance(en, np_int_type):
        po_en = en.item()
      else:
        assert isinstance(en, int)
      assert isinstance(po_en, int), "strange en type:" + str(type(po_en))
      seq.append([po_en])
    
  if ens_len >= 0:
    seq = seq[0:ens_len]
  return seq


def replace_unk_with_none_in_list(lls):
  seq = []
  for ll in lls:
    if isinstance(ll, list):
      slot = []
      seq.append(slot)
      for lll in ll:
        judge_and_append(slot, lll)
    else:
      judge_and_append(seq, ll)
  return seq


def judge_and_append(seq, ll):
  if 0 <= ll <= 2 or n_skt <= ll <= n_skt + 2:
    seq.append(None)
  else:
    seq.append(ll)


def get_unit_expand_sequence_list(ens_list, ens_len):
  res = []
  for ens in ens_list:
    r_ens = get_unit_expand_sequence(ens, ens_len, infer_stage=True)
    res.append(r_ens)
  return res
    



