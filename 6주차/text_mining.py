import re
from konlpy.tag import Komoran
import json

komoran = Komoran()


def text_cleaning(doc):
	# 한국어를 제외한 글자를 제거하는 함수
	# 한국어, 띄어쓰기 제외하고 제거
	doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
	return doc


def define_stopwords(path):

	SW = set()
	# 불용어를 추가하는 방법 1.
	# SW.add("있다.")

	# 불용어를 추가하는 방법 2.
	# stopwords-ko.txt 에 직접 추가

	with open(path, encoding = 'utf-8') as f:
		for word in f:
			SW.add(word)

	return SW


def text_tokenizing(doc):
	# 형태소 뽑아내기
	# list comprehension을 풀어서 쓴 코드

	return [word for word in komoran.morphs(doc) if word not in SW and len(word) > 1]



# json 파일을 불러와 content 값을 출력하는 코드

with open('test.json', 'r', encoding='UTF8') as f:
	json_data = json.load(f)

'''
content = json_data[0]['talk']['content']['HS01']
print(content)
text2 = content
'''


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


'''
list = [i['talk']['content']['HS01'] for i in json_data]
print(list)
'''

# text3 = """HS01": "진로가 너무 걱정스러워. 내가 진짜 뭘 해야 할지 모르겠어.", "SS01": "미래가 두려우시군요. 하고 싶은 일이 있으세요?", "HS02": "하고 싶은 일은 있는데 지금 시작하기에 너무 늦은 것 같아서 선뜻 도전을 못 하겠어.", "SS02": "도전하는 것에 두려움을 느끼고 계시네요.", "HS03": "", "SS03": """""

for i in list:
	SW = define_stopwords("stopwords-ko.txt")

	cleaned_text = text_cleaning(i)
	print("\n전처리 : ", cleaned_text)

	tokenized_text = text_tokenizing(cleaned_text)
	print("\n형태소 분석 : ", tokenized_text)
