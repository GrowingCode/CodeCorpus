import difflib
import os

filename1 = 'E:/code/witness_merges/2021-1-10/YYXWitnessSktPEMerges.json'
filename2 = 'E:/code/witness_merges/2021-1-13/YYXWitnessSktPEMerges.json'

with open(filename1) as f1,open(filename2) as f2:
  content1 = f1.read().splitlines(keepends=True)
  content2 = f2.read().splitlines(keepends=True)
  
d = difflib.HtmlDiff()
htmlcontent = d.make_file(content1, content2)
des_path = os.path.join(os.path.expanduser('~'),"Desktop")
with open(des_path + '/' + 'diff_res.html', 'w') as f:
  f.write(htmlcontent)

