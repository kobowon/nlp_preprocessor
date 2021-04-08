import os
from tqdm import tqdm

import pandas as pd

import re 
from soynlp.normalizer import repeat_normalize
import fasttext
import nltk
#문장 분절 패키지 (필요시 변경 가능)
from nltk.tokenize import sent_tokenize
import hanja

class LanguageModelPreprocessor:
    '''
    A processor for language model.
    '''
    def __init__(self, args):
        if not args.input_path:
            raise ValueError("전처리할 파일 경로를 입력하지 않았습니다")
        if not args.output_dir:
            raise ValueError("출력 디렉토리를 입력하지 않았습니다")
        
        #내부적으로 사용되는 변수 : 언더바 하나(_) : python은 다른 언어에 있는 접근 제어자(public, private, protected)가 없음, 모두 public
        self._model = fasttext.load_model(os.getcwd() + '/lang_detect_fasttext.ftz')
        self.input_path = args.input_path
        self.clean_sentences = []

    #한자 존재여부 판별
    def _detect_hanja(self, sentence):
        #generator to list
        tokens = [x for x in hanja.split_hanja(sentence)]
        if len(tokens) > 1:
            return True
        return False

    #짧은 문장 판별
    def _detect_short(self, sentence):
        tokens = sentence.split()
        if len(tokens) < 4:
            return True
        else:
            return False

    #전반적인 언어가 한국어인지 판별 (fasttext 언어모델 활용 : 비문 구별 능력은 미숙)
    def _detect_korean(self, sentence):
        if sentence is None:
            return False
        try:
            result = self._model.predict(sentence,k=1)
            main_lang = result[0][0]
            main_prob = result[1][0]

            if main_lang == '__label__ko':
                if float(main_prob) > 0.99:
                    return True
                else:
                    return False
            else: 
                return False
        except:
            return False

    #문장 전처리
    def _clean(self, text):
        #1.문자열 내의 영어, 일부 특수 문자제외하고 모두 제거
        text = re.sub(pattern=r'[^- .,?!:/@$%~％·a-zA-Z가-힣0-9]',repl='',string=text)
        #2. 반복되는 글자 제거
        text = repeat_normalize(text, num_repeats=2)
        return text

    #중복 문장 제거
    def _delete_duplication(self, sentences):
        unique_sentence = set(sentences)
        sentences = list(unique_sentence)
        return sentences

    #두 문장 이상으로 구성된 문단 분리
    def _separate_paragraph(self, paragraph):
        sentences = sent_tokenize(paragraph)
        result = [sentence.strip() for sentence in sentences]
        return result

    def preprocess(self):
        sentences = []
        with open(self.input_path,'r',encoding='utf-8') as f:
            #한 line에 paragraph가 있다고 가정 (문장이 잘 안 잘린 경우가 있다고 가정하고 한 번 더 분절)
            for paragraph in f:
                paragraph = paragraph.strip()
                #빈 문장
                if paragraph:
                    #1.일부 Brace로 감싸진 단어 제거 (뉴스의 경우 괄호 안 한자 다수 : 한자 문장 제거 전, 최대한의 데이터 확보를 위함)  
                    paragraph = re.sub(pattern=r'\[[^)]*\]|\([^)]*\)|\<[^)]*\>|\【[^)]*\】|\{[^)]*\}',repl='',string=paragraph)

                    #2.한자 존재 유무 판별
                    if self._detect_hanja(paragraph):
                        continue

                    #3.두 문장 이상이 붙어있는 경우 다 한 문장으로 분리
                    separate_sens = self._separate_paragraph(paragraph)
                    sentences.extend(separate_sens)

        clean_sentences = []
        for sentence in tqdm(sentences, mininterval=5):
            #4. 짧은 문장 제외
            if self._detect_short(sentence):
                continue
            if not self._detect_korean(sentence):
                continue
            #5. 문장 전처리
            clean_sen = self._clean(sentence)
            clean_sentences.append(clean_sen)

        #6. 중복 문장 제거
        self.clean_sentences = self._delete_duplication(clean_sentences)
        return self.clean_sentences
