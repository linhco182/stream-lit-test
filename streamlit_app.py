import streamlit as st
import tensorflow as tf
from tensorflow import keras
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel

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
  return predicted_classes

# load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model("./model_rev1.h5", custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})
    return model

with st.spinner("Loading Model...."):
    model=load_model()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

def main():
    st.title('Email Classifier')

    st.sidebar.title('About')
    st.sidebar.info(
        "This is a simple Streamlit app to classify emails into categories. "
        "You can paste the content of your email in the box below and click on 'Classify' button."
    )

    st.subheader('Email Body')
    email_content = st.text_area('Paste the content of your email in here:', height=200)

    if st.button('Classify'):
        if email_content:
            classification_result = classify_email(email_content)
            st.success(f'Classification result: {", ".join(classification_result)}')  # Join classes with comma
        else:
            st.warning('Please paste the content of your email before classifying.')

if __name__ == "__main__":
    main()
