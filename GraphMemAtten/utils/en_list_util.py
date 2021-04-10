from meta_info.non_hyper_constant import all_skeleton_each_id_str


def skt_each_en_to_string(en):
  en_str = all_skeleton_each_id_str[en]
  return en_str

def skt_each_en_list_to_string(en_ll):
  a_str = ""
  for en in en_ll:
    en_str = all_skeleton_each_id_str[en]
    a_str += en_str + "$"
  return a_str
  



