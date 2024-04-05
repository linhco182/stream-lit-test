import streamlit as st
import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
import subprocess
import os
from streamlit_gsheets import GSheetsConnection
import pandas as pd

def update_csv(email_content, labels, existing_data):
    new_row = pd.DataFrame(
      [
        {
          "text":email_content,
          "label_1":labels[0],
          "label_2":labels[1],
          "label_3":labels[2]
        }
      ]
    )
    updated_df = pd.concat([existing_data, new_row], ignore_index = True)
    conn.update(worksheet="email_entries", data = updated_df)


def encode_email(email_content):
  encoded = tokenizer.encode_plus(email_content, truncation=True, max_length=256, padding='max_length', return_tensors="tf")
  test_encoded_dict = {key: val for key, val in encoded.items()}
  return test_encoded_dict


# Function to classify email content
def classify_email(email_content):
  encoded = encode_email(email_content)
  prob = model.predict(encoded)[0]
  # Define threshold
  threshold = 0.5

  # Convert probabilities to binary predictions based on threshold
  binary_predictions = tf.cast(prob >= threshold, dtype=tf.int32)
  classes = ['Request for Meeting', 'Request for Action', 'Request for Information']
  predicted_classes = [classes[i] for i in range(len(classes)) if binary_predictions[i] == 1]
  if not predicted_classes:
        return ['Delivery of Information']
  return predicted_classes

# load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():

    if not os.path.isfile('model.h5'):
        subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/linhco182/stream-lit-test/master/model_rev1.h5"'], shell=True)
    model=tf.keras.models.load_model("model.h5", custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})
    return model

with st.spinner("Loading Model...."):
    model=load_model()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    conn=st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet="email_entries", usecols=list(range(4)), ttl=5)
    existing_data = existing_data.dropna(how='all')

def main():
    main = st.secrets
    st.title('Email Classifier')

    st.sidebar.title('About')
    st.sidebar.write("You can paste the content of your email in the box below and click on 'Classify' button.")
    st.sidebar.write("It will sort the email to using the following classifiers:")
    st.sidebar.write("**1. Request for Meeting** e.g. a casual ask for coffee catchup, finalising details of a scheduled meeting")
    st.sidebar.write("**2. Request for Action** e.g. asking for recipient to review work, 'can you print this out?'")
    st.sidebar.write("**3. Request for Information** e.g. Technical Request, 'can I have the name of the representative...'")
    st.sidebar.write("**4. Delivery of Information** e.g. when the email is purely used to convey information or opinion")

    st.sidebar.info("Any information submitted will be used to further improve the model. Any feedback regarding the program, such as inclusion of new tags or different way to sort emails, please send to linhcobui182@gmail.com")


    st.info("Please refrain from sharing sensitive information.")
    st.write("<<< Information regarding app stored in sidebar")
    st.write("Data will not be explicitly stored unless the 'Submit Data for Save' button is pressed; however, it is important to note that data transfer occurs in an unencrypted manner.")
    st.subheader('Email Body')
    email_content = st.text_area('Paste the content of your email in here:', height=200)
    if st.button('Classify'):
        if email_content:
            classification_result = classify_email(email_content)
            st.success(f'Classification result: {", ".join(classification_result)}')  # Join classes with comma
        else:
            st.warning('Please paste the content of your email before classifying.')

    st.subheader('Submit Data')
    # Checkbox for "Request for Meeting"
    request_for_meeting = st.checkbox('Request for Meeting')

    # Checkbox for "Request for Action"
    request_for_action = st.checkbox('Request for Action')

    # Checkbox for "Request for Information"
    request_for_information = st.checkbox('Request for Information')

    useless = st.checkbox('Delivery of Information')

    # Button to submit data
    if st.button('Submit data for save'):
      if email_content:
        # Encode labels based on checkbox states
        labels = [int(request_for_meeting), int(request_for_action), int(request_for_information)]
        # Update CSV file
        update_csv(email_content, labels, existing_data)
        st.success('Data saved successfully!')
      else:
        st.warning('Please paste the content of your email before submitting data.')




if __name__ == "__main__":
    main()
