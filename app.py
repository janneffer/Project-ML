import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load datasets
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
treatment = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv("dataset/medications.csv")

# Load model
svc = pickle.load(open('model/svc.pkl', 'rb'))

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    tre = treatment[treatment['disease'] == dis]['Treatment']

    return desc, pre, med, tre

# Symptoms dictionary and diseases list
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    # Tambahkan semua gejala lainnya...
}
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 
    # Tambahkan semua penyakit lainnya...
}

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Streamlit app
st.title("Disease Prediction App")
st.write("Pilih gejala Anda untuk mengetahui prediksi penyakit!")

# Form untuk memilih gejala
selected_symptoms = st.multiselect("Pilih gejala:", options=list(symptoms_dict.keys()))

if st.button("Prediksi"):
    if not selected_symptoms:
        st.warning("Harap pilih setidaknya satu gejala.")
    else:
        predicted_disease = get_predicted_value(selected_symptoms)
        dis_des, precautions, medications, treatment = helper(predicted_disease)

        # Tampilkan hasil
        st.success(f"Penyakit yang diprediksi: {predicted_disease}")
        st.write("**Deskripsi Penyakit:**", dis_des)
        st.write("**Pencegahan:**")
        st.write("\n".join([f"- {prec}" for prec in precautions[0]]))
        st.write("**Pengobatan yang disarankan:**")
        st.write(", ".join(medications))
        st.write("**Perawatan yang disarankan:**", treatment)

