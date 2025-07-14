import vertexai
from vertexai.generative_models import GenerativeModel, Image
import Levenshtein
from unidecode import unidecode

from glob import glob
import cv2
import json
import time
from image import read_rg_dataset

def ocr_iou(duples):
  total_length = 0
  total_errors = 0
  for g, p in duples:
      if g == "*****":
          continue
      total_length += len(g)
      total_errors += min(Levenshtein.distance(unidecode(g.upper()), unidecode(p.upper())),
                          len(g))

  if total_length == 0:
      return 0
  metric = 1 - (total_errors/total_length)

  return max(metric, 0)

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

examples = generate_examples()

regs = {
  "rg": 'Registro Geral',
  "nome": 'Nome',
  "filiacao1": "Filiação 1",
  "filiacao2": "Filiação 2",
  "datanasc": "Data de Nascimento",
  "orgaoexp": "Órgão de Expedição",
  "rh": "Fator RH",
  "naturalidade": "Naturalidade",
  "obs": "Observação",
  "serial": "Código Serial",
  "codsec": "Código de Controle",
  "te": "T. Eleitor",
  "ctps": "CTPS",
  "serie": "Série",
  "uf": "UF",
  "dataexp": "Data de Expedição",
  "dni": "DNI",
  "cpf": "CPF",
  "rg": "Registro Geral (RG)",
  "regcivil": "Registro Civil",
  "regcivil1": "Registro Civil",
  "regcivil2": "Registro Civil",
  "pis": "NIS/PIS/PASEP",
  "militar": "Cert. Militar",
  "cnh": "CNH",
  "cns": "CNS",
  "profissional": "Identidade Profissional"
}

run_name = "rtmdet_warp_v1_gemini_1.5"
wait_time = 30
ims, lbs = read_rg_dataset(transform=True, image_files_only=False,
                            prediction_box=True, pred_dir="rtmpreds/",
                            save_in_aux=True, image_dir="padded/")
#ims, lbs = read_rg_dataset("synthetic", transform=False, image_files_only=True, prediction_box=False,
#                             image_dir="warped_icip/")
cnt = 0
cnt_entities = 0

r = 0
cnt = 0

project_id = "valued-network-443319-k9"
vertexai.init(project=project_id, location="us-central1")
model = GenerativeModel(model_name="gemini-1.5-flash-002") # previous: "gemini-1.5-flash-002"

def make_prompt(image, labels, example, ask="entities"):
  if ask == "entities":
    ks = list(labels.keys())
    prompt = "Leia este documento e responda: qual é o " + ", ".join([regs[x] for x in ks[:-1]]) +  \
                                                                f" e {regs[ks[-1]]}?"
    prompt += "\n" + "Responda no formato JSON, com uma lista de elementos do tipo {\"chave\": \"valor\"}, " + \
              "onde as chaves correspondem aos campos requisitados. Caso haja mais de uma linha no valor, " + \
              "retorne uma lista com todas as linhas."

    inp = [image, prompt]
    if example is not None:
      imex = Image.load_from_file("synthetic/base_warped_images/" + example['fname'])
      promptex = example['output']  
      inp += ["\n\nUm exemplo de documento e resposta vem a seguir:\n", imex, promptex]
  elif ask == "corners":
    prompt = "Here is a picture of a document. Identify the four corners of the document and give me" + \
              " their coordinates in the image in a JSON format."
    inp = [image, prompt]
  return inp


def get_response(model, image, labels, example = None):
  inp = make_prompt(image, labels, example=example, ask="entities")
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

failures = []
auxs = list(glob("aux_im/*"))
flx = glob("gemini_outputs/rtmdet_warp_v1_gemini_1.5/*")
flx = ["aux_im\\" + x.split("\\")[-1][:-4] for x in flx]
fls = [x.replace("\\", "/") for x in auxs if x not in flx]

print(len(fls))

for ifn,lb in zip(ims,lbs):
  if ifn not in fls:
    continue
  im = Image.load_from_file(ifn)
  root = ifn.split("\\")[-1]
  if root in fronts:
    ex = examples['front1'] if root != examples['front1']['fname'] else examples['front2']
  else:
    ex = examples['back1'] if root != examples['back1']['fname'] else examples['back2']

  try:
    r = get_response(model, im, lb['regions'], example=None)
  except Exception as e:
    print(f"File {ifn} failed with error:", e)
    failures.append(ifn)
    time.sleep(wait_time)
    continue
  #r = get_response(model, im, lb['regions'], example=ex)
  print(ifn, r)

  root = ifn.split("\\")[-1].split("/")[-1]
  nf = f"gemini_outputs/{run_name}/{root}.txt"
  with open(nf, "w", encoding='utf-8') as fd:
    fd.write(r)
  time.sleep(wait_time)

print("Failures:", failures)
