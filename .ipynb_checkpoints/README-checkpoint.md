# NLP Preprocessor
##### TMAX A&C, AE1-3, BowonKo

T-Preprocessor is a processor for NLP preprocessing.

Processor type
- Language model preprocessor
- 추가 예정
 
---

## Arguments
Arguments for language model preprocessing (preprocess.py)

| argument            | type    | description                                                  | default                                      |
| ------------------- | ------- | ------------------------------------------------------------ | -------------------------------------------- |
| `--input_path`       | `str`   | 전처리할 파일 경로 | `None`                                          |
| `--preprocessor_type`     | `str`   | 전처리기 유형     | `lm`                                          |
| `--output_dir`        | `str`   | 전처리된 문장을 저장할 디렉토리                         | `None`                        

## Installation
필요한 패키지 설치

```sh
pip install -r requirements.txt
```

전처리 수행

```sh
python preprocess.py \
  --input_path ./test_data/210406_고등학교_국어.txt \
  --preprocessor_type lm \
  --output_dir ./result/
```