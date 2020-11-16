from konlpy.tag import Okt
import json

okt=Okt()

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
print(list)

for i in list:
    print(okt.morphs(i))
