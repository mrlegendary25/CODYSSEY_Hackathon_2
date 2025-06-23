from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from authlib.integrations.flask_client import OAuth
import requests
import google.generativeai as genai
from datetime import datetime
import os
import re
from supabase import create_client, Client
import uuid
import logging
import pickle
import tensorflow as tf
import time
import subprocess
import tempfile
import sys
import numpy as np
import json
import pandas as pd
import ast
import itertools
from difflib import SequenceMatcher  # Add to imports
import os


symptoms_binary=[]


import json



symptoms_list= [
    'anxiety and nervousness',
    'depression',
    'depressive or psychotic symptoms',
    'emotional symptoms',
    'hostile behavior',
    'abusing alcohol',
    'drug abuse',
    'fainting',
    'feeling ill',
    'excessive anger',
    'disturbance of memory',
    'delusions or hallucinations',
    'temper problems',
    'fears and phobias',
    'low self-esteem',
    'hysterical behavior',
    'obsessions and compulsions',
    'antisocial behavior',
    'nightmares'
]

from flask_cors import CORS
app = Flask(__name__)
app.secret_key = os.urandom(24) 

# Initialize OAuth
oauth = OAuth(app)


# Initialize Gemini
genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('gemini-pro')

# Configure Google OAuth
google = oauth.register(


)

# Initialize Supabase
supabase = create_client(


)


@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login/google')
def google_login():
    redirect_uri = url_for('google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_authorize():
    token = google.authorize_access_token()
    resp = google.get('https://www.googleapis.com/oauth2/v3/userinfo')
    user_info = resp.json()
    
    email = user_info['email']
    name = user_info['name']
    
    # Check if user exists in Supabase
    result = supabase.table('users').select('*').eq('email', email).execute()

    
    if len(result.data) == 0:
        # New user, create account
        new_user = {
            'email': email,
            'name': name,
            'created_at': datetime.utcnow().isoformat()
        }
        result = supabase.table('users').insert(new_user).execute()
        user = result.data[0]
    else:
        user = result.data[0]
    
    # Store user in session
    session['user'] = user
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

from datetime import datetime

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get user's chat history
    chats = supabase.table('chat_sessions')\
        .select('*')\
        .eq('user_id', session['user']['id'])\
        .order('last_message_time', desc=True)\
        .execute()
    
    # Convert timestamp strings to datetime objects
    for chat in chats.data:
        if chat['last_message_time']:
            chat['last_message_time'] = datetime.fromisoformat(chat['last_message_time'].replace('Z', '+00:00'))
    
    return render_template('dashboard.html', chats=chats.data)



def get_chat_context(session_id):
    """Retrieve chat context from previous message summaries"""
    response = supabase.table('messages')\
        .select('message_summary')\
        .eq('session_id', session_id)\
        .order('created_at', desc=True)\
        .limit(5)\
        .execute()
    
    context = " ".join([msg['message_summary'] for msg in response.data])
    return context

def analyze_message(message, is_user_message=True):
    """Generate summary and sentiment for a message"""
    prompt = f"""
    Analyze the following {'user message' if is_user_message else 'bot response'}:
    "{message}"
    
    Provide:
    1. A 10-15 word summary
    2. A 1-3 word sentiment (e.g., anxious, hopeful, concerned)
    
    Format: summary|||sentiment
    """
    
    response = model.generate_content(prompt)
    summary, sentiment = response.text.strip().split('|||')
    return summary.strip(), sentiment.strip()


@app.route('/chat/<session_id>')
def chat(session_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get chat session details
    chat_session = supabase.table('chat_sessions')\
        .select('*')\
        .eq('id', session_id)\
        .single()\
        .execute()
    
    if not chat_session.data:
        return redirect(url_for('dashboard'))
    
    # Get chat messages
    messages = supabase.table('messages')\
        .select('*')\
        .eq('session_id', session_id)\
        .order('created_at')\
        .execute()
    
    return render_template('chat.html', 
                         chat_session=chat_session.data,
                         messages=messages.data)

@app.route('/start_chat', methods=['POST'])
def start_chat():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    mode = data.get('mode')
    topic = data.get('topic')
    
    # Generate a title based on mode and topic
    if mode == 'general':
        title = 'General Chat'
    elif mode == 'topic':
        title = f'{topic.title()} Discussion'
    else:
        title = 'Symptom Assessment'
    
    # Create new chat session
    session_id = str(uuid.uuid4())
    chat_session = {
        'id': session_id,
        'user_id': session['user']['id'],
        'mode': mode,
        'topic': topic,
        'title': title
    }
    
    supabase.table('chat_sessions').insert(chat_session).execute()
    
    # Generate and store initial bot message
    if mode == 'general':
        initial_message = "Hi! I'm here to listen and chat with you. How are you feeling today?"
    elif mode == 'topic':
        initial_message = f"I understand you'd like to discuss {topic}. What's on your mind?"
    else:
        initial_message = "I'll help you explore your symptoms. Please describe what you're experiencing."
    
    summary, sentiment = analyze_message(initial_message, False)
    supabase.table('messages').insert({
        'session_id': session_id,
        'is_user': False,
        'full_message': initial_message,
        'message_summary': summary,
        'sentiment': sentiment
    }).execute()
    
    return jsonify({
        'session_id': session_id
    })


@app.route('/delete_chat/<session_id>', methods=['DELETE'])
def delete_chat(session_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # First verify the chat belongs to the user
    chat = supabase.table('chat_sessions')\
        .select('*')\
        .eq('id', session_id)\
        .eq('user_id', session['user']['id'])\
        .single()\
        .execute()
    
    if not chat.data:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Delete messages first (due to foreign key constraint)
    supabase.table('messages')\
        .delete()\
        .eq('session_id', session_id)\
        .execute()
    
    # Then delete the chat session
    supabase.table('chat_sessions')\
        .delete()\
        .eq('id', session_id)\
        .execute()
    
    return jsonify({'success': True})


# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_input_symptoms(symptoms_list):
    """Validate that input symptoms are in the allowed list"""
    try:
        if not isinstance(symptoms_list, list):
            raise ValueError("Input symptoms must be a list")
        if not all(isinstance(s, (int, float)) for s in symptoms_list):
            raise ValueError("All symptoms must be numeric (0 or 1)")
        if len(symptoms_list) != 377:
            raise ValueError(f"Expected 377 symptoms, got {len(symptoms_list)}")
        return True
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        raise

def load_model_and_encoder():
    """Load the model and label encoder with enhanced error handling"""
    try:
        model_path = os.path.join('model', 'Model_4_better.h5')
        encoder_path = os.path.join('model', 'label_encoder.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found at: {encoder_path}")
            
        # Load model and encoder
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info("Successfully loaded model and encoder")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading model or encoder: {str(e)}")
        raise

def load_drug_data():
    """Load drug-related CSV files with error handling"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, 'data')
        
        # Load CSV files
        finaldiseases = pd.read_csv(os.path.join(data_dir, 'finaldiseases.csv'))
        druginteraction = pd.read_csv(os.path.join(data_dir, 'druginteractionsfinal.csv'))
        singledrugeffect = pd.read_csv(os.path.join(data_dir, 'singledrugsideeffect.csv'))
        finaldiseases = finaldiseases.fillna('')  # Or dropna() or use mean/median
        druginteraction = druginteraction.fillna('')
        singledrugeffect = singledrugeffect.fillna('')
        
        # Validate data
        if finaldiseases.empty or druginteraction.empty or singledrugeffect.empty:
            raise ValueError("One or more CSV files are empty")
            
        logger.info("Successfully loaded drug data")
        return finaldiseases, singledrugeffect, druginteraction
    except Exception as e:
        logger.error(f"Error loading drug data: {str(e)}")
        raise

def get_unique_medicines(prediction_list, finaldiseases):
    """Get unique medicines for predicted diseases"""
    try:
        results = []
        for disease in prediction_list:
            drug_info = finaldiseases[finaldiseases['disease'] == disease]['drug'].values
            if len(drug_info) > 0:
                actual_drug_list = ast.literal_eval(drug_info[0])
                if isinstance(actual_drug_list, list):
                    unique_drugs = list(set(actual_drug_list))
                    results.append((disease, unique_drugs))
        return results
    except Exception as e:
        logger.error(f"Error getting unique medicines: {str(e)}")
        raise

def get_first_5_medicines(result_list):
    """Get first 5 medicines for each disease"""
    try:
        return [(disease, medicines[:5]) for disease, medicines in result_list]
    except Exception as e:
        logger.error(f"Error getting first 5 medicines: {str(e)}")
        raise

def get_side_effects_for_medicines(all_medicines, drug_df):
    """Get side effects for medicines"""
    try:
        medicine_side_effects = {}
        for medicine in all_medicines:
            row = drug_df[drug_df['drug_name'].str.lower() == medicine.lower()]
            if not row.empty:
                medicine_side_effects[medicine] = row.iloc[0]['side_effects']
            else:
                row = drug_df[drug_df['generic_name'].str.lower() == medicine.lower()]
                if not row.empty:
                    medicine_side_effects[medicine] = row.iloc[0]['side_effects']
        return medicine_side_effects
    except Exception as e:
        logger.error(f"Error getting side effects: {str(e)}")
        raise

def get_interactions_for_pairs(all_medicines, interaction_df):
    """Get drug interactions for medicine pairs"""
    try:
        interaction_dict = {}
        for drug1, drug2 in itertools.combinations(all_medicines, 2):
            pair_key = f"{drug1}|||{drug2}"
            
            pair1 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug1.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug2.lower())
            ]
            pair2 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug2.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug1.lower())
            ]
            
            if not pair1.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug1,
                    'drug_b': drug2,
                    'interaction': pair1.iloc[0]['Interaction'],
                    'risk_level': pair1.iloc[0]['Risk_Level']
                }
            elif not pair2.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug2,
                    'drug_b': drug1,
                    'interaction': pair2.iloc[0]['Interaction'],
                    'risk_level': pair2.iloc[0]['Risk_Level']
                }
        return interaction_dict
    except Exception as e:
        logger.error(f"Error getting drug interactions: {str(e)}")
        raise


SYMPTOMS_FILE = 'symptoms_detect.json'

def fetch_chat_logs(session_id):
    response = supabase.table("messages").select("full_message").eq("session_id", session_id).order("created_at", desc=False).execute()
    messages = [msg["full_message"] for msg in response.data] if response.data else []
    return messages

def analyze_text_with_gemini(text):
    symptoms = [
        'anxiety and nervousness',
        'depression',
        'depressive or psychotic symptoms',
        'emotional symptoms',
        'hostile behavior',
        'abusing alcohol',
        'drug abuse',
        'fainting',
        'feeling ill',
        'excessive anger',
        'disturbance of memory',
        'delusions or hallucinations',
        'temper problems',
        'fears and phobias',
        'low self-esteem',
        'hysterical behavior',
        'obsessions and compulsions',
        'antisocial behavior',
        'nightmares'
        ]
    prompt = f"""
    Analyze this patient conversation to identify psychological symptoms from this list:
    {symptoms}

    Guidelines:
    1. Look for both explicit mentions and implicit indications
    2. Consider emotional states, behavioral patterns, and cognitive descriptions
    3. Match to the closest symptom even if exact words aren't used
    4. Rate confidence for each symptom (0-1)

    Example Analysis:
    "I've been avoiding social situations" → fears and phobias (0.85)
    "I drink a bottle of whiskey every night" → abusing alcohol (0.95)
    "Sometimes I see things others don't" → delusions or hallucinations (0.9)

    Conversation:
    {text}

    Provide results as:
    [SYMPTOM FROM LIST] (confidence score)
    Separate multiple symptoms with new lines
    """
    
    try:
        response = model.generate_content(prompt)
        detected_symptoms = []

        # Parse Gemini response
        for line in response.text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Extract symptom name and confidence
            match = re.match(r"^(.+?)\s*\((\d+\.?\d*)\)\)?$", line)
            if match:
                symptom_name = match.group(1).strip().lower()
                confidence = float(match.group(2))
                
                # Find closest match in symptoms list
                for official_symptom in symptoms:
                    if SequenceMatcher(
                        None, 
                        symptom_name, 
                        official_symptom.lower()
                    ).ratio() > 0.8:
                        if confidence >= 0.6:  # Confidence threshold
                            detected_symptoms.append(official_symptom)
                        break

        # Remove duplicates while preserving order
        seen = set()
        detected_symptoms = [s for s in detected_symptoms 
                           if not (s in seen or seen.add(s))]
        # Create binary array matching original symptoms list
        tot_symp = ['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness']
        symptoms_binary = [1 if s in detected_symptoms else 0 
                          for s in tot_symp]

        # Store in session temporarily
        session['auto_symptoms'] = {
            'binary': symptoms_binary,
            'symptoms': detected_symptoms
        }
        
        data = {
        'symptoms_binary': symptoms_binary,
        'detected_symptoms': detected_symptoms
        }

# Write to JSON file
        with open('static/symptoms_data.json', 'w') as f:
            json.dump(data, f, indent=4)
        
        print(detected_symptoms)
        print(symptoms_binary)

        return detected_symptoms
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed'}), 500
    
def convert_to_binary_array(detected_symptoms):
    # List of all possible symptoms (should match the list used in port 3000)
    all_symptoms = ['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness']  # ... rest of symptoms list
    # Add all 377 symptoms

    # Initialize binary array with zeros
    binary_array = [0] * len(all_symptoms)
    
    # Set 1 for detected symptoms
    for symptom in detected_symptoms:
        if symptom in all_symptoms:
            binary_array[all_symptoms.index(symptom)] = 1
    
    return binary_array

@app.route("/analyze_chat", methods=["POST"])
def analyze_chat():
    data = request.json
    session_id = data.get("session_id")


    # Get chat messages from Supabase
    chat_logs = fetch_chat_logs(session_id)

    if not chat_logs:
        return jsonify({"success": False, "error": "No chat logs found."})

    # Analyze using Gemini API
    detected_symptoms = analyze_text_with_gemini(" ".join(chat_logs))

    # Save detected symptoms
    with open("symptoms_detected.json", "w") as f:
        json.dump({"session_id": session_id, "symptoms": detected_symptoms }, f)

    return jsonify({"success": True, "detected_symptoms": detected_symptoms})


@app.route('/symptoms_detected.json')
def check_symptoms():
    try:
        if os.path.exists(SYMPTOMS_FILE):
            with open(SYMPTOMS_FILE, 'r') as f:
                data = json.load(f)
            os.remove(SYMPTOMS_FILE)  # Delete after reading
            return jsonify(data)
        return jsonify({'error': 'No symptoms found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/chat/<session_id>/messages', methods=['POST'])
def send_message(session_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    user_message = data['message']
    
    # Analyze and store user message
    user_summary, user_sentiment = analyze_message(user_message, True)
    supabase.table('messages').insert({
        'session_id': session_id,
        'is_user': True,
        'full_message': user_message,
        'message_summary': user_summary,
        'sentiment': user_sentiment
    }).execute()
    
    # Get chat context
    context = get_chat_context(session_id)
    
    # Get chat session details for context
    chat_session = supabase.table('chat_sessions')\
        .select('*')\
        .eq('id', session_id)\
        .single()\
        .execute()
    
    # Generate bot response using context and session info
    mode = chat_session.data['mode']
    topic = chat_session.data['topic']
    
    prompt = f"""
    Chat mode: {mode}
    {f'Topic: {topic}' if topic else ''}
    Previous context: {context}
    
    User message: {user_message}
    
    You are a mental health support chatbot. Respond empathetically and supportively while keeping the context and chat mode in mind.
    """
    
    response = model.generate_content(prompt)
    bot_message = response.text.strip()
    
    # Analyze and store bot response
    bot_summary, bot_sentiment = analyze_message(bot_message, False)
    supabase.table('messages').insert({
        'session_id': session_id,
        'is_user': False,
        'full_message': bot_message,
        'message_summary': bot_summary,
        'sentiment': bot_sentiment
    }).execute()
    
    return jsonify({'message': bot_message})


@app.route('/send-symptoms', methods=['POST'])
def send_symptoms():
    try:
        # Get symptoms_binary from your request
        symptoms_binary = request.json.get('symptoms_binary')
        
        # Send to port 3000 without waiting for response
        requests.post('http://localhost:3000/receive-symptoms', 
                     json={'symptoms': symptoms_binary})
        
        return jsonify({'status': 'sent'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_graphs/<session_id>', methods=['GET'])
def generate_graphs(session_id):
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Get all user messages from this chat session
        messages = supabase.table('messages')\
            .select('full_message')\
            .eq('session_id', session_id)\
            .eq('is_user', True)\
            .execute()
        print(messages)
        
        if not messages.data:
            return jsonify({'error': 'No messages found'}), 404
        
        # Combine all messages into one string
        combined_messages = " . ".join([msg['full_message'] for msg in messages.data])
        print(combined_messages)
        # Create a temporary file to store the messages
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(combined_messages)
            temp_path = temp_file.name
        
        try:
            # Get the absolute path to app2.py
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # app2_path = os.path.join(current_dir, 'app2.py')
            current_dir = 'C:\\Users\\Prashant\\Documents\\Himnish\\Web Development\\C-ODYSSEY'
            app2_path = 'C:\\Users\\Prashant\\Documents\\Himnish\\Web Development\\C-ODYSSEY\\app2.py'
            command = [sys.executable, app2_path, '--text', temp_path]
            print(f"Running command: {' '.join(command)}")
            # Run app2.py with the temporary file
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # Set a timeout of 30 seconds
                cwd=current_dir  # Set working directory
            )
            print('lol not working')
            # Log the output for debugging
            app.logger.debug(f"stdout: {result.stdout}")
            app.logger.debug(f"stderr: {result.stderr}")
            
            if result.returncode != 0:
                raise Exception(f"app2.py failed: {result.stderr}")
            
            # Parse the JSON output from app2.py
            try:
                graphs_data = json.loads(result.stdout)
                return jsonify(graphs_data)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse JSON output: {e}. Output was: {result.stdout}")
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                app.logger.error(f"Failed to delete temporary file: {e}")
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis took too long to complete'}), 500
    except Exception as e:
        app.logger.error(f"Error generating graphs: {str(e)}")
        return jsonify({'error': f'Failed to generate graphs: {str(e)}'}), 500

@app.route('/mediscope')
def mediscopes():
    return render_template('mediscope.html', 
                         symptoms_list=symptoms,
                         symptoms_binary=symptoms_binary)


# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# List of symptoms (377 symptoms)
symptoms = ['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness']
def validate_input_symptoms(symptoms_list):
    """Validate that input symptoms are in the allowed list"""
    try:
        if not isinstance(symptoms_list, list):
            raise ValueError("Input symptoms must be a list")
        if not all(isinstance(s, (int, float)) for s in symptoms_list):
            raise ValueError("All symptoms must be numeric (0 or 1)")
        if len(symptoms_list) != 377:
            raise ValueError(f"Expected 377 symptoms, got {len(symptoms_list)}")
        return True
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        raise

def load_model_and_encoder():
    """Load the model and label encoder with enhanced error handling"""
    try:
        model_path = os.path.join('model', 'Model_4_better.h5')
        encoder_path = os.path.join('model', 'label_encoder.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found at: {encoder_path}")
            
        # Load model and encoder
        model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info("Successfully loaded model and encoder")
        return model, label_encoder
    except Exception as e:
        logger.error(f"Error loading model or encoder: {str(e)}")
        raise

def load_drug_data():
    """Load drug-related CSV files with error handling"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, 'data')
        
        # Load CSV files
        finaldiseases = pd.read_csv(os.path.join(data_dir, 'finaldiseases.csv'))
        druginteraction = pd.read_csv(os.path.join(data_dir, 'druginteractionsfinal.csv'))
        singledrugeffect = pd.read_csv(os.path.join(data_dir, 'singledrugsideeffect.csv'))
        finaldiseases = finaldiseases.fillna('')  # Or dropna() or use mean/median
        druginteraction = druginteraction.fillna('')
        singledrugeffect = singledrugeffect.fillna('')
        
        # Validate data
        if finaldiseases.empty or druginteraction.empty or singledrugeffect.empty:
            raise ValueError("One or more CSV files are empty")
            
        logger.info("Successfully loaded drug data")
        return finaldiseases, singledrugeffect, druginteraction
    except Exception as e:
        logger.error(f"Error loading drug data: {str(e)}")
        raise

def get_unique_medicines(prediction_list, finaldiseases):
    """Get unique medicines for predicted diseases"""
    try:
        results = []
        for disease in prediction_list:
            drug_info = finaldiseases[finaldiseases['disease'] == disease]['drug'].values
            if len(drug_info) > 0:
                actual_drug_list = ast.literal_eval(drug_info[0])
                if isinstance(actual_drug_list, list):
                    unique_drugs = list(set(actual_drug_list))
                    results.append((disease, unique_drugs))
        return results
    except Exception as e:
        logger.error(f"Error getting unique medicines: {str(e)}")
        raise

def get_first_5_medicines(result_list):
    """Get first 5 medicines for each disease"""
    try:
        return [(disease, medicines[:5]) for disease, medicines in result_list]
    except Exception as e:
        logger.error(f"Error getting first 5 medicines: {str(e)}")
        raise

def get_side_effects_for_medicines(all_medicines, drug_df):
    """Get side effects for medicines"""
    try:
        medicine_side_effects = {}
        for medicine in all_medicines:
            row = drug_df[drug_df['drug_name'].str.lower() == medicine.lower()]
            if not row.empty:
                medicine_side_effects[medicine] = row.iloc[0]['side_effects']
            else:
                row = drug_df[drug_df['generic_name'].str.lower() == medicine.lower()]
                if not row.empty:
                    medicine_side_effects[medicine] = row.iloc[0]['side_effects']
        return medicine_side_effects
    except Exception as e:
        logger.error(f"Error getting side effects: {str(e)}")
        raise

def get_interactions_for_pairs(all_medicines, interaction_df):
    """Get drug interactions for medicine pairs"""
    try:
        interaction_dict = {}
        for drug1, drug2 in itertools.combinations(all_medicines, 2):
            pair_key = f"{drug1}|||{drug2}"
            
            pair1 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug1.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug2.lower())
            ]
            pair2 = interaction_df[
                (interaction_df['Drug_A'].str.lower() == drug2.lower()) & 
                (interaction_df['Drug_B'].str.lower() == drug1.lower())
            ]
            
            if not pair1.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug1,
                    'drug_b': drug2,
                    'interaction': pair1.iloc[0]['Interaction'],
                    'risk_level': pair1.iloc[0]['Risk_Level']
                }
            elif not pair2.empty:
                interaction_dict[pair_key] = {
                    'drug_a': drug2,
                    'drug_b': drug1,
                    'interaction': pair2.iloc[0]['Interaction'],
                    'risk_level': pair2.iloc[0]['Risk_Level']
                }
        return interaction_dict
    except Exception as e:
        logger.error(f"Error getting drug interactions: {str(e)}")
        raise



@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Log request
        logger.info("Received prediction request")
        
        # Get and validate input data
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({"error": "Missing symptoms data"}), 400
            
        # Validate input format
        input_symptoms = np.array([data['symptoms']])
        validate_input_symptoms(data['symptoms'])

        # Load required components
        model, label_encoder = load_model_and_encoder()
        finaldiseases, singledrugeffect, druginteraction = load_drug_data()

        # Make prediction
        predicted_probabilities = model.predict(input_symptoms)
        sorted_indices = np.argsort(predicted_probabilities[0])[::-1]
        max_prob = predicted_probabilities[0][sorted_indices[0]]
        
        # Process predictions
        prediction_list = []
        prediction_list_prob = []
        extra_prediction = None
        extra_prediction_prob = None
        
        predicted_diseases = label_encoder.inverse_transform(sorted_indices)
        
        # Handle predictions based on probability threshold
        if max_prob < 0.5:
            first_prob = predicted_probabilities[0][sorted_indices[0]]
            prediction_list.append(predicted_diseases[0])
            prediction_list_prob.append(float(first_prob))
            
            for i in range(1, min(3, len(sorted_indices))):
                next_prob = predicted_probabilities[0][sorted_indices[i]]
                if first_prob - next_prob <= 0.2:
                    prediction_list.append(predicted_diseases[i])
                    prediction_list_prob.append(float(next_prob))
        else:
            first_prob = predicted_probabilities[0][sorted_indices[0]]
            prediction_list.append(predicted_diseases[0])
            prediction_list_prob.append(float(first_prob))
            
            for i in range(1, len(sorted_indices)):
                next_prob = predicted_probabilities[0][sorted_indices[i]]
                if first_prob - next_prob <= 0.2:
                    prediction_list.append(predicted_diseases[i])
                    prediction_list_prob.append(float(next_prob))
                else:
                    break
        
        # Get extra prediction
        extra_idx = len(prediction_list)
        if extra_idx < len(predicted_diseases):
            extra_prediction = predicted_diseases[extra_idx]
            extra_prediction_prob = float(predicted_probabilities[0][sorted_indices[extra_idx]])

        # Get medicine recommendations
        result = get_unique_medicines(prediction_list, finaldiseases)
        result_extra = get_unique_medicines([extra_prediction] if extra_prediction else [], finaldiseases)
        
        result_first5 = get_first_5_medicines(result)
        result_extra_first5 = get_first_5_medicines(result_extra)

        # Get all medicines
        all_medicines = []
        logger.info("Starting Flask app")
        for disease, medicines in result_first5 + result_extra_first5:
            all_medicines.extend(medicines)
        all_medicines = list(set(all_medicines))

        # Get side effects and interactions
        side_effects_dict = get_side_effects_for_medicines(all_medicines, singledrugeffect)
        interaction_results = get_interactions_for_pairs(all_medicines, druginteraction)

        # Format predictions
        formatted_predictions = [
            {
                "disease": disease,
                "probability": round(float(prob), 4)
            } 
            for disease, prob in zip(prediction_list, prediction_list_prob)
        ]

        # Format extra prediction
        formatted_extra = None
        if extra_prediction:
            formatted_extra = {
                "disease": extra_prediction,
                "probability": round(float(extra_prediction_prob), 4)
            }

        # Prepare response
        response = {
            "predictions": formatted_predictions,
            "extra_prediction": formatted_extra,
            "medicines": {
                "main": [
                    {
                         "disease": disease,
                        "medications": medicines
                    }
                    for disease, medicines in result_first5
                ],
                "extra": [
                    {
                        "disease": disease,
                        "medications": medicines
                    }
                    for disease, medicines in result_extra_first5
                ]
            },
            "side_effects": side_effects_dict,
            "interactions": interaction_results
        }
        logger.info("Starting Flask app")
        
        logger.info("Successfully generated prediction response")
        return jsonify(response)


    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(debug=True)
