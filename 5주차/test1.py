import json

# json 파일을 불러와 content 값을 출력하는 코드

with open('test.json', 'r', encoding='UTF8') as f:
	json_data = json.load(f)

# content 부분을 list로 만들어 출력하는 코드
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