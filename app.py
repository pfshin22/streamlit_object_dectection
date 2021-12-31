import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import date, datetime
import numpy as np

from object_detection_app import run_object_detection



# 디렉토리 정보와 파일을 알려주면, 해당 디렉토리에
# 파일을 저장하는 함수를 만들겁니다.
def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))


def main():
    st.title('Tensorflow Object Detection')

    menu = ['Object Detection', 'About']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == 'Object Detection' :

        # 파일 업로드 코드 작성. 카피 앤 페이스트 해서 사용하세요.
        image_file = st.file_uploader("이미지를 업로드 하세요", type=['png','jpg','jpeg'])
        if image_file is not None :
            # 프린트문은 디버깅용으로서, 터미널에 출력한다.
            print(type(image_file))
            print(image_file.name)
            print(image_file.size)
            print(image_file.type)

            # 파일명을, 현재시간의 조합으로 해서 만들어보세요.
            # 현재시간.jpg
            current_time = datetime.now()
            print(current_time)
            print(current_time.isoformat().replace(':', '_'))
            current_time = current_time.isoformat().replace(':', '_')
            image_file.name = current_time + '.jpg'

            # 파일을 저장할 수 있도록, 위의 함수를 호출하자.
            # save_uploaded_file('temp', image_file)

            # 오브젝트 디텍션을 여기서 한다.            
            img = Image.open(image_file)

            img = np.array(img)
            # 넘파이 어레이를 오브젝트 디텍션 함수에 넘겨준다.
            run_object_detection(img)


    

if __name__ == '__main__' :
    main()