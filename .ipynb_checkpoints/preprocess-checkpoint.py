import os
import argparse
from preprocessor import LanguageModelPreprocessor

preprocessors = {
        "lm":LanguageModelPreprocessor,
}    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path",type=str, default=None,help="전처리할 파일 경로")
    parser.add_argument("--preprocessor_type", type=str, default="lm", help="전처리기 유형")
    parser.add_argument("--output_dir", type=str, default="지정안함", help="전처리된 문장을 저장할 디렉토리 ")
    
    args = parser.parse_args()
    
    preprocessor = preprocessors[args.preprocessor_type](args)
    sentences = preprocessor.preprocess()
    
    output_path = '/'.join([args.output_dir,os.path.basename(args.input_path)])
    
    #txt 파일 저장 line by line
    with open(output_path, 'w', encoding='utf-8') as f:
        for sen in sentences:
            f.write(sen+'\n')

    #데이터 확인
    with open(output_path,'r',encoding='utf-8') as f:
        sentences = f.read().split('\n')
        
        for i in sentences[:20]:
            print(i)
        print('데이터 총 수 : ',len(sentences))


if __name__ == "__main__":
    main()