import streamlit as st
import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
import subprocess
import os
import pandas as pd

def update_csv(email_content, labels):
    if not os.path.isfile('data.csv'):
      # Create a new DataFrame
      data_df = pd.DataFrame(columns=['Email Content', 'Request for Meeting', 'Request for Action', 'Request for Information'])
      # Write the DataFrame to data.csv
      new_row = {'Email Content': email_content, 'Request for Meeting': labels[0], 
               'Request for Action': labels[1], 'Request for Information': labels[2]}
      data_df = pd.concat([data_df, pd.DataFrame([new_row])], ignore_index=True)
      data_df.to_csv('data.csv', index=False)
    else:
      data_df = pd.read_csv('data.csv')
      new_row = {'Email Content': email_content, 'Request for Meeting': labels[0], 
               'Request for Action': labels[1], 'Request for Information': labels[2]}
      data_df = pd.concat([data_df, pd.DataFrame([new_row])], ignore_index=True)
      data_df.to_csv('data.csv', index=False)


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

def main():
    st.title('Email Classifier')

    st.sidebar.title('About')
    st.sidebar.info(
        "This is a simple app to classify emails into categories. "
        "You can paste the content of your email in the box below and click on 'Classify' button."
        "It will classify the email to one of 4 classes: 1. Request for Meeting 2. Request for Action 3. Request for Information 4. Purely a Delivery of Information."
    )

    st.subheader('Email Body')
    st.write("Please refrain from sharing sensitive information. While data is not actively stored, please be aware that data transfer occurs in an unencrypted manner.")
    email_content = st.text_area('Paste the content of your email in here:', height=200)
    if st.button('Classify'):
        if email_content:
            classification_result = classify_email(email_content)
            st.success(f'Classification result: {", ".join(classification_result)}')  # Join classes with comma
        else:
            st.warning('Please paste the content of your email before classifying.')

    # Checkbox for "Request for Meeting"
    request_for_meeting = st.checkbox('Request for Meeting')
    
    # Checkbox for "Request for Action"
    request_for_action = st.checkbox('Request for Action')
    
    # Checkbox for "Request for Information"
    request_for_information = st.checkbox('Request for Information')

    # Button to submit data
    if st.button('Submit data for save'):
        # Get email content
        email_content = st.text_input('Email Content', '')
    
        # Encode labels based on checkbox states
        labels = [int(request_for_meeting), int(request_for_action), int(request_for_information)]
    
        # Update CSV file
        update_csv(email_content, labels)
        st.success('Data saved successfully!')

    

if __name__ == "__main__":
    main()
