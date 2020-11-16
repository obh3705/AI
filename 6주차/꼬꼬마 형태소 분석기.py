from konlpy.tag import Kkma
import json

with open('test.json', 'r', encoding='UTF8') as f:
	json_data = json.load(f)

list = []
for i in json_data:
    json_txt = list.append(i['talk']['content']['HS01'])
    json_txt = list.append(i['talk']['content']['SS01'])
    json_txt = list.append(i['talk']['content']['HS02'])
    json_txt = list.append(i['talk']['content']['SS02'])
    json_txt = list.append(i['talk']['content']['HS03'])
    json_txt = list.append(i['talk']['content']['SS03'])

for item in list:
    if item == "":
        list.remove("")

kkma=Kkma()

for i in list:
    print("1. kkma.morphs: ", kkma.morphs(i), "\n")
    print("2. kkma.pos: ", kkma.pos(i), "\n")
    print("3. kkma.nouns: ", kkma.nouns(i), "\n")