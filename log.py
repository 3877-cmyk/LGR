import joblib
ld_mod= joblib.load('log.pkl')

import streamlit as st

st.title("logistic regression model")
st.write("Enter the number of hours studied:")
input_feature= st.number_input("Enter a Numerical value:", min_value=0, max_value=24)

stat= ld_mod.predict([[input_feature]])

if stat == 1:
    st.write("Pass")

else:
    st.write("Fail")