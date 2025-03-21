#https://gri.jp/media/entry/3798
#回帰分析評価指標
#https://di-acc2.com/analytics/ai/8852/
#全体のまとめ
#https://qiita.com/shuhigashi/items/0fb37468e64c76f4b245
from cgi import test
import streamlit as st
import pandas as pd
import datetime
from pycaret.classification import load_model, predict_model
#from pycaret.regression import load_model, predict_model
import pickle
import streamlit.components.v1 as stc
from PIL import Image





st.subheader('Texture情報から転移ありなしの予測デモ機！')



st.write('')
st.write('')
st.write('')


#ファイルがアップロード
uploaded_files = st.file_uploader("予測したいdata；CSVをアップロード", accept_multiple_files= False)

if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df = df.drop(df.columns[0],axis=1)
    st.dataframe(df)
    st.write('')


    final_model = load_model('Texturetest')
    
    #dia_test = pd.read_csv('C:\\Users\\koya\\Desktop\\test.csv',encoding='shift_jis')
    #dia_test = dia_test.drop(dia_test.columns[0],axis=1)

    new_prediction = predict_model(final_model, data=df)

    st.subheader('label:予測結果')
    st.table(new_prediction)
    
    
    csv = new_prediction.to_csv().encode('utf-8')

    # 処理結果をCSVとしてダウンロード
    st.download_button(
        label="処理結果をダウンロード",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )
    


