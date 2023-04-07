import streamlit as st
from catboost import CatBoostClassifier, Pool
import pandas as pd

cb = CatBoostClassifier(n_estimators=200,
                        loss_function ='MultiClass',
                        learning_rate = 0.4,
                        depth=3,
                        task_type = 'CPU',
                        random_state = 1,
                        verbose = True)

cb.load_model('cb_model.json') 


def predict (Diplome, Ville, Exp_label ,technos1 ,technos2,technos3,technos4,technos5,technos6,technos7):
    prediction = cb.predict(pd.DataFrame([[Diplome, Ville, Exp_label ,technos1 ,technos2,technos3,technos4,technos5,technos6,technos7]], 
                                         columns=['Diplome','Ville','Exp_label','technos1','technos2','technos3','technos4','technos5','technos6','technos7']))
    return prediction


st.title('Data profiles Predictor')
st.image('cover.PNG')
st.header('Enter the characteristics of the profile:')
Entreprise = st.text_input('Entreprise ')
#Diplome = st.text_input('Diplome: ')
Diplome = st.selectbox('Diplome: ', ('No diploma', 'Bachelor', 'Master', 'PhD'))
#Ville = st.text_input('Ville: ')
Ville = st.selectbox('Ville: ', ('Paris', 'Lyon', 'Marseille', 'Toulouse', 'Lille', 'Bordeaux', 'Nantes', 'Rennes', 'Rouen', 'Strasbourg', 'Toulon', 'Nice', 'Grenoble', 'Montpellier'))
#Exp_label = st.text_input('Exp_label: ')
Exp_label = st.selectbox('Exp_label: ', ('debutant', 'Confirme', 'Avance', 'Expert'))
technos1 = st.text_input('technos1: ')
technos2 = st.text_input('technos2: ')
technos3 = st.text_input('technos3: ')
technos4 = st.text_input('technos4: ')
technos5 = st.text_input('technos5: ')
technos6 = st.text_input('technos6: ')
technos7 = st.text_input('technos7: ')

if st.button('Predict Profile'):
    profile = predict(Diplome, Ville, Exp_label ,technos1 ,technos2,technos3,technos4,technos5,technos6,technos7)
    st.success(profile)
