import json
from utils.analysis_util import print_first_param_right_but_second_param_wrong


def read_analysis_info(a_dir):
  with open(a_dir + '/' + "z_stand_infer_skt_ens.json", 'r') as file_object:
    std_ifr = json.loads(file_object.read())
  with open(a_dir + '/' + "z_stand_oracle_skt_ens.json", 'r') as file_object:
    std_olc = json.loads(file_object.read())
  with open(a_dir + '/' + "z_multi_infer_skt_ens.json", 'r') as file_object:
    mlt_ifr = json.loads(file_object.read())
  with open(a_dir + '/' + "z_multi_oracle_skt_ens.json", 'r') as file_object:
    mlt_olc = json.loads(file_object.read())
  return std_ifr, std_olc, mlt_ifr, mlt_olc

first_dir = "C:/Users/yangy/Desktop/temp/e_zoot_run_info_record"
second_dir = "C:/Users/yangy/Desktop/temp/pe_zoot_run_info_record"

f_std_ifr, f_std_olc, f_mlt_ifr, f_mlt_olc = read_analysis_info(first_dir)
s_std_ifr, s_std_olc, s_mlt_ifr, s_mlt_olc = read_analysis_info(second_dir)

print_first_param_right_but_second_param_wrong(f_std_ifr, f_std_olc, s_std_ifr, s_std_olc)





