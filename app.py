
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# Define ResNet18WithDropout if not already defined

# Match the notebook's model definition
class ResNet18WithDropout(nn.Module):
    def __init__(self, base_model, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(base_model.fc.in_features, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# loading the model that we saved earlier
base_model = resnet18(pretrained=False)
model = ResNet18WithDropout(base_model, dropout_p=0.5)
model.load_state_dict(torch.load("pneumonia_resnet18 model.pth", map_location=torch.device('cpu')))
model.eval()

#defining the image transformations and preprocessing steps
transform = transforms.Compose([
	transforms.Resize((224, 224)),
    transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark mode toggle with icons and better layout
from streamlit.components.v1 import html
if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False

toggle_html = '''
<style>
.toggle-switch {
    display: flex;
    align-items: center;
    gap: 0.5em;
    margin-bottom: 1.2em;
}
.toggle-switch input[type=checkbox] {
    width: 40px;
    height: 22px;
    appearance: none;
    background: #c6c6c6;
    outline: none;
    border-radius: 20px;
    transition: background 0.3s;
    position: relative;
}
.toggle-switch input[type=checkbox]:checked {
    background: #232526;
}
.toggle-switch input[type=checkbox]::before {
    content: '';
    position: absolute;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    top: 2px;
    left: 2px;
    background: #fff;
    transition: 0.3s;
}
.toggle-switch input[type=checkbox]:checked::before {
    left: 20px;
    background: #232526;
}
.toggle-switch label {
    font-size: 1.1em;
    color: #2E86C1;
    font-weight: 600;
}
</style>
<div class="toggle-switch">
    <span style="font-size:1.3em;">🌞</span>
    <input id="darkmode-toggle" type="checkbox" onchange="window.parent.postMessage({isDark: this.checked}, '*')" {checked}>
    <span style="font-size:1.3em;">🌙</span>
    <label for="darkmode-toggle">Dark Mode</label>
</div>
<script>
    const iframe = window.frameElement;
    window.addEventListener('message', (event) => {
        if (event.data && typeof event.data.isDark === 'boolean') {
            window.parent.postMessage({isDark: event.data.isDark}, '*');
        }
    });
</script>
'''.replace('{checked}', 'checked' if st.session_state['dark_mode'] else '')

def update_dark_mode():
        st.session_state['dark_mode'] = not st.session_state['dark_mode']

sidebar_placeholder = st.sidebar.empty()
sidebar_placeholder.markdown(toggle_html, unsafe_allow_html=True)

# Listen for JS postMessage (Streamlit limitation: user must click twice to toggle, but this is the best workaround for now)
if st.sidebar.button('Apply Theme', key='theme_apply'):
        update_dark_mode()

st.sidebar.markdown("---")

# Sidebar with more info and style
st.sidebar.markdown(
    """
    <div style='text-align:center;'>
        <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOW1ZYnGjHtlfDgAqpcpe1U5eUIEJeGD_rsg&s' width='120'/>
    </div>
    <h2 style='color:#2E86C1; text-align:center;'>Pneumonia AI</h2>
    <hr style='border:1px solid #bbb;'>
    <p style='font-size:16px;'>
    <b>Upload a chest X-ray to detect pneumonia using a deep learning model.</b><br><br>
    <span style='color:#922B21;'>This tool is for educational and research purposes only. Not for clinical use.</span><br><br>
    <b>Developed by:</b> <span style='color:#117A65;'>SRIJA DE CHOWDHURY</span>
    </p>
    """,
    unsafe_allow_html=True
)

# Main header


# Custom background and main card with dark mode support
if st.session_state['dark_mode']:
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(120deg, #232526 0%, #414345 100%) !important;
            color: #f4f6f7 !important;
        }
        .main-card {
            background: #232526;
            color: #f4f6f7;
            border-radius: 18px;
            box-shadow: 0 4px 24px 0 rgba(46,134,193,0.10);
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            margin-bottom: 2rem;
        }
        .section-card {
            background: #2c2f34;
            color: #f4f6f7;
            border-radius: 12px;
            box-shadow: 0 2px 8px 0 rgba(46,134,193,0.07);
            padding: 1.5rem 1.5rem 1rem 1.5rem;
            margin-bottom: 1.5rem;
        }
        h1, h2, h4, b, .sidebar-content, .st-bb, .st-at, .st-af, .st-cg, .st-c6, .st-c5, .st-c4, .st-c3, .st-c2, .st-c1, .st-c0 {
            color: #f4f6f7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
        }
        .main-card {
            background: #fff;
            border-radius: 18px;
            box-shadow: 0 4px 24px 0 rgba(46,134,193,0.10);
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            margin-bottom: 2rem;
        }
        .section-card {
            background: #f8fafd;
            border-radius: 12px;
            box-shadow: 0 2px 8px 0 rgba(46,134,193,0.07);
            padding: 1.5rem 1.5rem 1rem 1.5rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if st.session_state['dark_mode']:
    st.markdown("""
    <div class='main-card'>
        <h1 style='text-align: center; color: #2E86C1; margin-bottom:0;'>🩺 Pneumonia Detection from Chest X-ray</h1>
        <h4 style='text-align: center; color: #117A65; margin-top:0;'>AI-powered Medical Imaging Assistant</h4>
        <hr style='border:1px solid #bbb;'>
        <div style='background-color: #232526; color: #f4f6f7; padding: 10px; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #444;'>
            <b>Instructions:</b> Upload a chest X-ray image (JPG/PNG). The AI will analyze and predict if pneumonia is present.<br>
            <b>Disclaimer:</b> This tool is not a substitute for professional medical advice.
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='main-card'>
        <h1 style='text-align: center; color: #2E86C1; margin-bottom:0;'>🩺 Pneumonia Detection from Chest X-ray</h1>
        <h4 style='text-align: center; color: #117A65; margin-top:0;'>AI-powered Medical Imaging Assistant</h4>
        <hr style='border:1px solid #bbb;'>
        <div style='background-color: #F9EBEA; color: #222; padding: 10px; border-radius: 10px; margin-bottom: 1.5rem;'>
            <b>Instructions:</b> Upload a chest X-ray image (JPG/PNG). The AI will analyze and predict if pneumonia is present.<br>
            <b>Disclaimer:</b> This tool is not a substitute for professional medical advice.
        </div>
    """, unsafe_allow_html=True)

# Example: Class distribution pie chart (static example, replace with real data if available)
st.markdown("<div class='section-card'><h4 style='color:#2E4053;'>Sample Data Distribution</h4>", unsafe_allow_html=True)
labels = ['Normal', 'Pneumonia']
sizes = [1500, 3500]  # Example values, replace with real counts if available
colors = ['#ABEBC6', '#F1948A']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
ax1.axis('equal')
st.pyplot(fig1)
st.markdown("</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    st.write("")
    st.write("<b>Classifying...</b>", unsafe_allow_html=True)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
        probs = torch.softmax(output, dim=1)
        threshold = 0.93
        prob_pneumonia = float(probs[0,1])
        prob_normal = float(probs[0,0])
        predicted = (prob_pneumonia > threshold)
        if predicted:
            st.markdown(
                f"<div style='background-color:#F1948A; padding:20px; border-radius:10px; text-align:center;'><h2 style='color:#922B21;'>🟥 Pneumonia Detected</h2></div>",
                unsafe_allow_html=True)
            st.progress(int(prob_pneumonia * 100))
        else:
            st.markdown(
                f"<div style='background-color:#ABEBC6; padding:20px; border-radius:10px; text-align:center;'><h2 style='color:#145A32;'>🟩 Normal</h2></div>",
                unsafe_allow_html=True)
            st.progress(int(prob_normal * 100))
        st.markdown(f"<b>Probability of Pneumonia:</b> <span style='color:#922B21'>{prob_pneumonia*100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Probability of Normal:</b> <span style='color:#145A32'>{prob_normal*100:.2f}%</span>", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("<h4 style='color:#2E4053;'>Prediction Probabilities</h4>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots()
        ax2.bar(['Normal', 'Pneumonia'], [prob_normal, prob_pneumonia], color=['#ABEBC6', '#F1948A'])
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Probability')
        for i, v in enumerate([prob_normal, prob_pneumonia]):
            ax2.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=12)
        st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
					  
		
    
    # Preprocess the image
	
