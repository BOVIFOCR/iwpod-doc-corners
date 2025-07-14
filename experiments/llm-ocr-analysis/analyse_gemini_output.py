import vertexai
from vertexai.generative_models import GenerativeModel, Image
import Levenshtein
from Levenshtein import distance as ldist
from unidecode import unidecode
from itertools import product

from glob import glob
import cv2
import json
import time
from image import read_rg_dataset

regs = {
  "Registro Geral": "rg",
  "rh": "rh",
  "SERIE": "serial",
  "UF": "uf",
  "filiacao": "filiacao1",
  "REGCIVIL": "reg-civil",
  "RG": "rg",
  "CPF": "cpf",
  "MILITAR": "militar",
  "CNH": "cnh",
  "PIS": "pis",
  "CTPS": "ctps",
  "TE": "te",
  "DATAEXP": "dataexp",
  "Número do Cartório": "regcivil",
  "Certidão de Nascimento": "regcivil",
  "rg": "rg",
  "cpf": "cpf",
  "Registro Civil2": "regcivil",
  "Cert. Nascimento Cartório": "regcivil",
  "Registro de Nascimento": "regcivil",
  "RG": "rg",
  "Nome": "Nome",
  "nome": "nome",
  "filiacao1": "filiacao1",
  "filiacao2": "filiacao2",
  "Filiação": "filiacao1",
  "Filiação 1": "filiacao1",
  "Filiação 2": "filiacao2",
  "registrocivil": "regcivil",
  "nis/pis/pasep": "pis",
  "te": "te",
  "dni": "dni",
  "uf": "uf",
  "serie": "serie",
  "cns": "cns",
  "nispispasep": "pis",
  "pis": "pis",
  "Órgão Expedição": "orgaoexp",
  "cnh": "cnh",
  "profissional": "profissional",
  "militar": "militar",
  "registro_civil": "regcivil",
  "regcivil2": "regcivil",
  "data_expedicao": "dataexp",
  "certmilitar": "militar",
  "identidadeprofissional": "profissional",
  "nis_pis_pasep": "pis",
  "registrocivil2": "regcivil",
  "CTPS / Série / UF": "uf",
  "ctps": "ctps",
  "regcivil": "regcivil",
  "Data de Nascimento": "datanasc",
  "Órgão Expedidor": "orgaoexp",
  "Órgão de Expedição": "orgaoexp",
  "dataexp": "dataexp",
  "Fator RH": "rh",
  "Naturalidade": "naturalidade",
  "Observação": "obs",
  "Código Serial": "serial", # And codsec
  "Código": "codsec",
  "Código DF": "codsec",
  "Código de Série": "codsec",
  "Número do RG": "codsec",
  "Número da Carteira": "codsec",
  "Código RG": "codsec",
  "Código de Controle": "codsec",
  "Código de Barras": "codsec",
  "Código de Identificação": "codsec",
  "Código de identificação": "codsec",
  "Código de Verificação": "codsec",
  "T. Eleitor": "te",
  "CTPS": "te",
  "Série": "serie",
  "UF": "uf",
  "Data de Expedição": "dataexp",
  "DNI": "dni",
  "CPF": "cpf",
  "Registro Geral (RG)": "rg",
  "Matrícula": "regcivil",
  "Número do Registro Civil": "regcivil", # and regcivil1, regcivil2
  "Livro Registro Civil": "regcivil", # and regcivil1, regcivil2
  "Registro Civil (localização)": "regcivil", # and regcivil1, regcivil2
  "Registro Civil": "regcivil", # and regcivil1, regcivil2
  "NIS/PIS/PASEP": "pis",
  "Cert. Militar": "militar",
  "CNH": "cnh",
  "datanasc": "dataanasc",
  "codsec": "codsec",
  "obs": "obs",
  "serial": "serial",
  "orgaoexp": "orgaoexp",
  "naturalidade": "naturalidade",
  "CNS": "cns",
  "Identidade Profissional": "profissional"
}

def read_gemini(fname):
  output = {}
  with open(fname, "r", encoding='utf-8') as fd:
    pd_js = json.loads(fd.read().split("```")[-2][4:])
  #if type(pd_js) == list and len(pd_js) == 1:
  #  pd_js = pd_js[0]
  try:
    if type(pd_js) == dict:
      ks = list(pd_js.keys())
      for k in ks:
        if k.startswith("Código") and k != "Código Serial":
          regs[k] = "codsec"
        output[regs[k]] = pd_js[k]
    elif type(pd_js) == list:
      for dct in pd_js:
        k = list(dct.keys())[0]
        v = dct[k]
        if 'chave' in dct.keys():
          k = dct['chave']
          v = dct['valor']
        elif k.startswith("Código") and k != "Código Serial":
          regs[k] = "codsec"
        output[regs[k]] = v
  except Exception as e:
    print(fname, pd_js, repr(e))
    exit(0)
  return output
      

def get_dist(s, t):
  return min(ldist(unidecode(s.upper()), unidecode(t.upper())), len(s))

def ocr_iou(duples):
  total_length = 0
  total_errors = 0
  for g, p in duples:
      if g == "*****" or p is None:
          continue
      total_length += len(g)
      total_errors += get_dist(g, p)

  if total_length == 0:
      return 0
  metric = 1 - (total_errors/total_length)

  return metric

def generate_examples():
  example_back = "10e63fbd-131a-470f-83ba-922574e2806a.jpg_1272412TzGsbye.jpg"
  back1 = cv2.imread(f"synthetic/base_warped_images/{example_back}")
  with open(f"synthetic/base_warped_labels/{example_back[:-4]}.json", "r", encoding='utf-8') as fd:
    back1lb = json.load(fd)['regions']
  back1rg = {k: v['text'] for k, v in back1lb.items()}
  back1out = json.dumps(back1rg, ensure_ascii=False, indent=2)

  example_back_2 = "049e93b8-49d9-41ee-ac34-37b53ca1dace.jpg_1613103XlYHthu.jpg"
  back2 = cv2.imread(f"synthetic/base_warped_images/{example_back_2}")
  with open(f"synthetic/base_warped_labels/{example_back_2[:-4]}.json", "r", encoding='utf-8') as fd:
    back2lb = json.load(fd)['regions']
  back2rg = {k: v['text'] for k, v in back2lb.items()}
  back2out = json.dumps(back2rg, ensure_ascii=False, indent=2)

  example_front = "00adf09c-37b9-4e1b-ac38-dec14390ccac.jpg_0616651dQaVMzE.jpg"
  front1 = cv2.imread(f"synthetic/base_warped_images/{example_front}")
  with open(f"synthetic/base_warped_labels/{example_front[:-4]}.json", "r", encoding='utf-8') as fd:
    front1lb = json.load(fd)['regions']
  front1rg = {k: v['text'] for k, v in front1lb.items()}
  front1out = json.dumps(front1rg, ensure_ascii=False, indent=2)

  example_front_2 = "018e3738-a0f7-4248-bb25-65998580543d.jpeg_0641023ZvDyGHh.jpg"
  front2 = cv2.imread(f"synthetic/base_warped_images/{example_front_2}")
  with open(f"synthetic/base_warped_labels/{example_front_2[:-4]}.json", "r", encoding='utf-8') as fd:
    front2lb = json.load(fd)['regions']
  front2rg = {k: v['text'] for k, v in front2lb.items()}
  front2out = json.dumps(front2rg, ensure_ascii=False, indent=2)

  return {
    "back1": {
      'fname': example_back,
      'image': back1,
      'output': back1out
    }, "back2": {
      'fname': example_back_2,
      'image': back2,
      'output': back2out
    }, "front1": {
      'fname': example_front,
      'image': front1,
      'output': front1out
    }, "front2": {
      'fname': example_front_2,
      'image': front2,
      'output': front2out
    },
  }

def match_regcivil(gt, preds):
  ls = list(product([0, 1], repeat=3))
  ret = preds[-1]
  for l in ls[2:]:
    next_try = "".join([x for i, x in enumerate(preds) if l[i] == 1])
    if get_dist(gt, next_try) < get_dist(gt, ret):
      ret = next_try
  return ret


do_ocr_gt = True
#run = "orig_run_v0" # no warping
#run = "gt_run_v1" # GT boxes
#run = "pred_run_v1" # IWPOD 
#run = "yolon_2000_v1" # YOLO
#run = "icip_warp_v1" # JDESKEW
run = "rtmdet_warp_v1_gemini_1.5" # RTMDET

ims, lbs = read_rg_dataset(transform=False, image_files_only=True)

cnt = 0
cnt_entities = 0

r = 0
cnt = 0

for im,lb in zip(ims, lbs):
  gt_js = lb['regions']
  root = im.split("\\")[-1]
  pdf2 = f"gemini_outputs/gt_run_v1/{root}.txt"

  if run == "rtmdet_warp_v1_gemini_1.5":
    root = root.split("_")[0].split(".")[0] + ".png"
  pdf = f"gemini_outputs/{run}/{root}.txt"
  try:
    pd_js = read_gemini(pdf)
    pd_js2 = read_gemini(pdf2)
  except:
    continue

  pds = {}
  if type(pd_js) == list:
    for i in pd_js:
      for k,v in i.items():
        pds[k] = v
  else:
    pds = pd_js

  pds2 = {}
  if type(pd_js2) == list:
    for i in pd_js2:
      for k,v in i.items():
        pds2[k] = v
  else:
    pds2 = pd_js2


  match = []
  for k,v in gt_js.items():
    if k not in pds.keys() or pds[k] is None:
      cnt_entities += 1
    elif v['text'] is not None:
      if type(pds[k]) == list:
        if len(pds[k]) > 1:
          pds[k] = match_regcivil(v['text'], pds[k])
        elif len(pds[k]) == 1:
          pds[k] = pds[k][0]
        else:
          pds[k] = ""
      if "\n" in pds[k]:
        pds[k] = pds[k].split("\n")
        pds[k] = pds[k][0] if ldist(pds[k][0], v['text']) < ldist(pds[k][1], v['text']) else pds[k][1]
      match.append((v['text'], pds[k]))
  #print(match, root)

  match2 = []
  for k,v in gt_js.items():
    if k not in pds2.keys() or pds2[k] is None:
      cnt_entities += 1
    elif v['text'] is not None:
      if type(pds2[k]) == list:
        if len(pds2[k]) > 1:
          pds2[k] = match_regcivil(v['text'], pds2[k])
        elif len(pds2[k]) == 1:
          pds2[k] = pds2[k][0]
        else:
          pds2[k] = ""
      if "\n" in pds2[k]:
        pds2[k] = pds2[k].split("\n")
        pds2[k] = pds2[k][0] if ldist(pds2[k][0], v['text']) < ldist(pds2[k][1], v['text']) else pds2[k][1]
      match2.append((v['text'], pds2[k]))
 
  second = 0 if do_ocr_gt else 1
  match3 = [(v1[1],v2[second]) for v1,v2 in product(match, match2) if v1[0] == v2[0]]

  ocr_metric = ocr_iou(match3)
  #print(ocr_metric, root)
  #if ocr_metric < 0.66:
  #  print(ocr_metric, root)
  cnt += 1
  r += ocr_metric
print(cnt, r/cnt)
#print(json.dumps(pds, indent=2))
#print(json.dumps({k:v['text'] for k, v in gt_js.items()}, indent=2))


exit(0)

project_id = "valued-network-443319-k9"
vertexai.init(project=project_id, location="us-central1")
model = GenerativeModel(model_name="gemini-1.5-flash-002")

def get_response(model, image, labels, example = None):
  ks = list(labels.keys())
  prompt = "Leia este documento e responda: qual é o " + ", ".join(ks[:-1]) + f" e {ks[-1]}?"
  prompt += "\n" + "Responda no formato JSON, com uma lista de elementos do tipo {\"chave\": \"valor\"}, " + \
            "onde as chaves correspondem aos campos requisitados. Caso haja mais de uma linha no valor, " + \
            "retorne uma lista com todas as linhas."
  inp = [image, prompt]
  if example is not None:
    imex = Image.load_from_file("synthetic/base_warped_images/" + example['fname'])
    promptex = example['output']  
    inp += ["\n\nUm exemplo de documento e resposta vem a seguir:\n", imex, promptex]
  
  r = model.generate_content(inp)
  return r.text

fronts = []
backs = []
with open("front_files.txt", 'r', encoding='utf-8') as fd:
  for l in fd.readlines():
    fronts.append(l.strip().split(" ")[0])
with open("back_files.txt", 'r', encoding='utf-8') as fd:
  for l in fd.readlines():
    backs.append(l.strip().split(" ")[0])

for ifn,lb in zip(ims[40:],lbs[40:]):
  im = Image.load_from_file(ifn)
  #gtf = ifn.split(".")[0] + ".json"
  #with open(gtf, "r", encoding='utf-8') as fd:
  #  js = json.load(fd)
  root = ifn.split("\\")[-1]
  if root in fronts:
    ex = examples['front1'] if root != examples['front1']['fname'] else examples['front2']
  else:
    ex = examples['back1'] if root != examples['back1']['fname'] else examples['back2']



  r = get_response(model, im, lb['regions'], example=ex)
  print(ifn, r)

  root = ifn.split("\\")[-1]
  nf = f"gemini_outputs/gt_run_v0/{root}.txt"
  with open(nf, "w", encoding='utf-8') as fd:
    fd.write(r)
  time.sleep(60)
  


#image_file = Image.load_from_file("output_docs/pred/000de277-9116-4a37-81b9-14ebe9dfc705.jpg")

#print(response)
#print(response.text)
