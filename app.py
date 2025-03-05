import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import emoji
import string
import streamlit as st

# Helper Functions
def preprocess_text(text):
    """Clean and preprocess the text data."""
    text = emoji.replace_emoji(text, '')
    text = ''.join(ch for ch in text if ch not in string.punctuation or ch in '.?')
    return ' '.join(text.split()).strip()

def parse_whatsapp_txt(file_content):
    """Parse WhatsApp chat from a txt file into a structured format."""
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[AaPpMm]{2})?)\s-\s([^:]+):\s(.+)'
    messages = []
    current_message = ''

    for line in file_content.decode('utf-8').splitlines():
        match = re.match(pattern, line.strip())
        if match:
            if current_message:
                messages.append(current_message)
            timestamp, sender, message = match.groups()
            if message.strip() == '<Media omitted>':
                current_message = ''
                continue
            current_message = {'timestamp': timestamp, 'sender': sender.strip(), 'message': preprocess_text(message.strip())}
        elif current_message:
            current_message['message'] += ' ' + preprocess_text(line.strip())
    if current_message:
        messages.append(current_message)
    return pd.DataFrame(messages) if messages else pd.DataFrame()

# Chatbot Class
class WhatsAppChatbot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.responses = {}
        self.common_responses = {}
        self.trained_messages = []

    def train(self, chat_df, target_user):
        """Train the chatbot on WhatsApp chat data for a specific user."""
        conversation_pairs = []
        labels = []

        for i in range(1, len(chat_df)):
            prev_message = chat_df.iloc[i-1]['message'].lower()
            current_message = chat_df.iloc[i]['message']
            current_sender = chat_df.iloc[i]['sender']

            if current_sender == target_user:
                conversation_pairs.append(prev_message)
                labels.append(current_message)
                self.trained_messages.append(prev_message)

                if prev_message in self.responses:
                    if current_message not in self.responses[prev_message]:
                        self.responses[prev_message].append(current_message)
                else:
                    self.responses[prev_message] = [current_message]

                self.common_responses[current_message] = self.common_responses.get(current_message, 0) + 1

        if conversation_pairs:
            self.X = self.vectorizer.fit_transform(conversation_pairs)
            y = [1] * len(labels)
            self.classifier.fit(self.X, y)
        else:
            st.warning(f"Warning: No messages from {target_user} found for training.")

    def respond(self, message):
        """Generate a response to a given message."""
        message = preprocess_text(message.lower())
        if message in self.responses:
            return np.random.choice(self.responses[message])
        try:
            message_vector = self.vectorizer.transform([message])
            similarities = cosine_similarity(message_vector, self.X)[0]
            max_similarity = np.max(similarities)
            if max_similarity > 0.3:
                best_match_idx = np.argmax(similarities)
                best_match = self.trained_messages[best_match_idx]
                return np.random.choice(self.responses[best_match])
            for trained_message in self.responses:
                if any(word in trained_message.split() for word in message.split()):
                    return np.random.choice(self.responses[trained_message])
            if self.common_responses:
                return max(self.common_responses.items(), key=lambda x: x[1])[0]
            return "I don’t have enough data to respond properly."
        except (ValueError, AttributeError):
            return "Hmm, I’m not sure what to say to that."

# Streamlit App Layout
st.set_page_config(page_title="WhatsApp Chatbot", page_icon="💬")
st.title("💬 WhatsApp Chatbot")
st.markdown("Upload a WhatsApp chat file (.txt) to train a chatbot that mimics your conversation style!")

# Initialize Session State
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_value' not in st.session_state:
    st.session_state.input_value = ""  # Placeholder for text input

# File Upload Section
uploaded_file = st.file_uploader("Upload your WhatsApp chat file", type="txt")
if uploaded_file is not None:
    chat_df = parse_whatsapp_txt(uploaded_file.read())
    if not chat_df.empty:
        st.subheader("Available Senders")
        senders = chat_df['sender'].unique()
        st.write(senders)

        # Sender Selection
        target_user = st.selectbox("Select the user to mimic:", senders, key="target_user")
        if st.button("Train Chatbot"):
            st.session_state.chatbot = WhatsAppChatbot()
            st.session_state.chatbot.train(chat_df, target_user)
            st.success("Chatbot trained successfully! Start chatting below.")
    else:
        st.error("No valid messages found in the file. Please check the file format.")

# Chat Interface
if st.session_state.chatbot:
    st.subheader("Chat with Your Bot")
    # Use placeholder variable instead of direct widget key manipulation
    user_input = st.text_input("Type your message:", value=st.session_state.input_value, key="user_input_field")
    if st.button("Send") and user_input:
        response = st.session_state.chatbot.respond(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.session_state.input_value = ""  # Reset input value
        # Rerun to clear the field
        st.rerun()

    # Display Chat History
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"{target_user}(**Bot**): {chat['bot']}")
            st.markdown("---")

# Footer
st.markdown("<hr><p style='text-align: center;'>Built by <a href=\"https://github.com/rohansadaphule\"><b><i>Rohan Sadaphule</i></b></a> with 🧠 using Streamlit</p>", unsafe_allow_html=True)