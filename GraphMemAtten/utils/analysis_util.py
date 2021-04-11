from utils.en_list_util import skt_each_en_to_string, skt_each_en_list_to_string
from utils.print_util import pretty_print_dict


def print_first_param_right_but_second_param_wrong(first_infer, first_oracle, second_infer, second_oracle):
  pass
  f_right_dict, f_wrong_dict, f_right_pe_right_dict, f_right_pe_wrong_dict, f_wrong_pe_wrong_dict = statistic_acc(first_infer, first_oracle)
  s_right_dict, s_wrong_dict, s_right_pe_right_dict, s_right_pe_wrong_dict, s_wrong_pe_wrong_dict = statistic_acc(second_infer, second_oracle)
  f_keys = set()
  s_keys = set()
  f_keys.update(f_right_dict.keys())
  f_keys.update(f_wrong_dict.keys())
  s_keys.update(s_right_dict.keys())
  s_keys.update(s_wrong_dict.keys())
  assert f_keys == s_keys
  for f_k in f_keys:
    f_size = f_right_dict.get(f_k, 0) + f_wrong_dict.get(f_k, 0)
    s_size = s_right_dict.get(f_k, 0) + s_wrong_dict.get(f_k, 0)
    assert f_size == s_size
  for f_k in f_keys:
    fr = f_right_dict.get(f_k, 0)
    sr = s_right_dict.get(f_k, 0)
    if fr > sr:
      print("=== " + f_k + " f_better ===")
      pretty_print_dict(f_right_pe_right_dict.get(f_k, {}))
      pretty_print_dict(f_right_pe_wrong_dict.get(f_k, {}))
      pretty_print_dict(f_wrong_pe_wrong_dict.get(f_k, {}))
      print("=== " + f_k + " s_worser ===")
      pretty_print_dict(s_right_pe_right_dict.get(f_k, {}))
      pretty_print_dict(s_right_pe_wrong_dict.get(f_k, {}))
      pretty_print_dict(s_wrong_pe_wrong_dict.get(f_k, {}))
  

def statistic_acc(a_infer, a_oracle):
  right_dict = {}
  wrong_dict = {}
  right_pe_right_dict = {}
  right_pe_wrong_dict = {}
  wrong_pe_wrong_dict = {}
  for raw_ai, ao in zip(a_infer, a_oracle):
#     print("len(ai):" + str(len(ai)) + "#len(ao):" + str(len(ao)))
    assert len(raw_ai) == 1
    ai = raw_ai[0]
    assert len(ai) == len(ao), "strange length, len(ai):" + str(len(ai)) + "," + str(type(ai)) + "#len(ao):" + str(len(ao)) + "," + str(type(ao))
    for sep_ai, sep_ao in zip(ai, ao):
      islist1 = isinstance(sep_ai, list)
      islist2 = isinstance(sep_ao, list)
      assert islist1 == islist2
      islist = islist1 and islist2
      if islist:
        sai_size = len(sep_ai)
        sao_size = len(sep_ao)
        list_str = skt_each_en_list_to_string(sep_ao)
        list_right = True
        if sai_size == sao_size:
          for i in range(sao_size):
            o_en = sep_ao[i]
            a_en = sep_ai[i]
            if a_en == o_en:
              pass
            else:
              list_right = False
              break;
        else:
          list_right = False
        for i in range(sao_size):
          o_en = sep_ao[i]
          o_en_str = skt_each_en_to_string(o_en)
          if i < sai_size:
            a_en = sep_ai[i]
            if a_en == o_en:
              right_dict[o_en_str] = right_dict.get(o_en_str, 0) + 1
              if list_right:
                pe_dict = right_pe_right_dict.get(o_en_str, {})
                pe_dict[list_str] = pe_dict.get(list_str, 0) + 1
                right_pe_right_dict[o_en_str] = pe_dict
              else:
                pe_dict = right_pe_wrong_dict.get(o_en_str, {})
                pe_dict[list_str] = pe_dict.get(list_str, 0) + 1
                right_pe_wrong_dict[o_en_str] = pe_dict
            else:
              wrong_dict[o_en_str] = wrong_dict.get(o_en_str, 0) + 1
              pe_dict = wrong_pe_wrong_dict.get(o_en_str, {})
              pe_dict[list_str] = pe_dict.get(list_str, 0) + 1
              wrong_pe_wrong_dict[o_en_str] = pe_dict
          else:
            wrong_dict[o_en_str] = wrong_dict.get(o_en_str, 0) + 1
            pe_dict= wrong_pe_wrong_dict.get(o_en_str, {})
            pe_dict[list_str] = pe_dict.get(list_str, 0) + 1
            wrong_pe_wrong_dict[o_en_str] = pe_dict
#         if list_right:
#           pass
#         else:
#           wrong_pe_dict[list_str] = wrong_pe_dict.get(list_str, 0) + 1
      else:
        o_en_str = skt_each_en_to_string(sep_ao)
        if sep_ai == sep_ao:
          right_dict[o_en_str] = right_dict.get(o_en_str, 0) + 1
        else:
          wrong_dict[o_en_str] = wrong_dict.get(o_en_str, 0) + 1
  return right_dict, wrong_dict, right_pe_right_dict, right_pe_wrong_dict, wrong_pe_wrong_dict
  
  
  


