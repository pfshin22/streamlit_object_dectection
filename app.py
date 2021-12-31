import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import date, datetime
import numpy as np

from object_detection_app import run_object_detection

import matplotlib.pyplot as plt



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

    choice = st.sidebar.selectbox('select the menu', menu)

    if choice == 'Object Detection' :

        selected_radio = st.radio('모델을 선택하세요', ('라인차트', '아리아차트', '바차트'))    
        if selected_radio == '라인차트' :
            chart_data = pd.DataFrame(df_corr[selected_corr])

            st.line_chart(chart_data)

        elif selected_radio == '아리아차트' :
            chart_data = pd.DataFrame(df_corr[selected_corr])

            st.area_chart(chart_data)

        elif selected_radio == '바차트' :
            chart_data = pd.DataFrame(df_corr[selected_corr])

            st.bar_chart(chart_data)
        # 유저가 컬럼을 선택하지 않은 경우 

        else : 
            st.write('선택한 컬럼이 없습니다.')

        # 파일 업로드 코드 작성. 카피 앤 페이스트 해서 사용하세요.
        image_file = st.file_uploader("Upload the image please", type=['png','jpg','jpeg'])
        if image_file is not None :

            min_score = st.slider('Set the lower limit of bounding number please : ', 1, 100, value=50)

            st.write('{} 이상의 detecting 확률로 설정합니다.' .format(min_score))

            # 프린트문은 디버깅용으로서, 터미널에 출력한다.
            # print(type(image_file))
            # print(image_file.name)
            # print(image_file.size)
            # print(image_file.type)

            # 파일명을, 현재시간의 조합으로 해서 만들어보세요.
            # 현재시간.jpg
            current_time = datetime.now()
            print(current_time)
            print(current_time.isoformat().replace(':', '_'))
            current_time = current_time.isoformat().replace(':', '_')
            image_file.name = current_time + '.jpg'

            # 파일을 저장할 수 있도록, 위의 함수를 호출하자.
            # save_uploaded_file('temp', image_file)

            st.caption('추출된 이미지 : ')
            # 오브젝트 디텍션을 여기서 한다.            
            img = Image.open(image_file)

            img = np.array(img)
            # 넘파이 어레이를 오브젝트 디텍션 함수에 넘겨준다.
            run_object_detection(img, min_score)

            st.success('Object dectection complete!')

            st.write('### 분석')
            
            st.write('### 바운딩 박스 히스토그램')
            fig, ax = plt.subplots()
            ax.hist(min_score, bins=10, range=(0.0,1.0))
            st.pyplot(fig)



    

if __name__ == '__main__' :
    main()