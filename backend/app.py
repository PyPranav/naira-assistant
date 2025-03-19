from flask import Flask, render_template
from flask_socketio import SocketIO
from flask import Flask, redirect, request, url_for, jsonify, send_from_directory

import io
import os
import json
import re
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from config import getFinalPrompt
from config import GROQ_TOOLS
from langchain.prompts import ChatPromptTemplate
import nltk
from youtubesearchpython import VideosSearch
import webbrowser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from PIL import Image
from bs4 import BeautifulSoup
from googlesearch import search
import GPUtil
from datetime import datetime

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil
import yfinance as yf
from groq import Groq
import time
import base64


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
ANSWERS = []

load_dotenv(find_dotenv('.env'))

NEXUS_URL = 'https://ollama.pypranav.com/api/generate'

CONTEXT = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

client = Groq(
    api_key= os.environ.get("GROQ_API_KEY"),
)





@socketio.on('newUser')
def handle_message(data):
    print('received id: ' + data)

@app.route('/')
def chat():
    return render_template('index.html')

@socketio.on('queryGroq')
def query(userQuery, context):
    global CONTEXT
    print('recieved', userQuery)

    CONTEXT = [z for z in context]

    MODEL = 'llama-3.1-8b-instant'

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant and your name is Naira answer kindly and seem super interested in the user, make sure to use your name when ever needed."
        }
    ]

    for z in context:
        messages.append(z)

    # messages.append(
    #     {
    #         "role": "user",
    #         "content": userQuery,
    #     })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=GROQ_TOOLS,
        tool_choice="auto",
        max_tokens=4096
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    i=0
    print(response_message.content, tool_calls)
    print()

    while response_message.content and ('function=' in response_message.content or 'function(' in response_message.content or '>{' in response_message.content):
        if i==10:
            break
        i+=1
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
            max_tokens=4096
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        print(i)

    print(response_message.content, tool_calls)
    print()

    if tool_calls:
        available_functions = {
            "search_web_and_answer": search_web_and_answer,
            "ask_llama": ask_llama,
            "send_email": send_email,
            "get_system_info": get_system_info,
            "youtube_search": youtube_search,
            "google_search": google_search,
            "generate_image": generateImage,
            "writeCode":writeCode,
            "open_app": open_app,
            'open_meet_and_send_mail':open_meet_and_send_mail
            # "createMeet": createMeet
        }

        socketio.emit('new', response_message.content)
        try:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                print({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
        except Exception:
            socketio.emit('new', 'Something went wrong')

        # second_response = client.chat.completions.create(
        #     model=MODEL,
        #     messages=messages
        # )
        # final_response = second_response.choices[0].message.content
    else:
        # response = client.chat.completions.create(
        #     model=MODEL,
        #     messages=messages,
        #     max_tokens=4096
        # )
        socketio.emit('new', response_message.content)

        
    #     final_response = response.choices[0].message.content
    # socketio.emit('new', final_response)



@socketio.on('query')
def query(userQuery, context):
    global CONTEXT
    print('recieved', userQuery)

    CONTEXT = [z for z in context]


    promptTemplate = ChatPromptTemplate.from_template(getFinalPrompt(userQuery))
    prompt = promptTemplate.format(userQuery=userQuery)

    headers = {
    "Content-Type": "application/json"
    }

    data = {
        "model": 'nexusraven',
        'prompt': prompt,
        'stream': False,
        "options":{"stop":['<bot_end>']}
    }

    response = requests.post(NEXUS_URL, headers=headers, data=json.dumps(data))
    print(response)
    

    response_text=''
    if response. status_code == 200:
        response_text = response.text
        data = json.loads( response_text)
        actual_response = data['response']
        response_text = actual_response.strip('Call:').strip()
    else:
        print( "Error:" ,response.status_code,response.text )

    socketio.emit('new', response_text)
    execute_code(response_text)
        # execute_code('search_web_and_answer("Latest anthorpic ai models")')

        # print(userId, userQuery)
           
        # socketio.emit('new', 'done')




@socketio.on('rag')
def rag(userQuery, rag, chats):
    global CONTEXT
    print("RAG!!!!", userQuery)
    CONTEXT = [z for z in chats]

    # socketio.emit('new', response_text)
    ask_llama(userQuery, rag)


def generateImage(query):
    socketio.emit('new', f'''Generating Image for '{query}'...''')

    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": "Bearer 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"}
    response = requests.post(API_URL, headers=headers, json=query)
    # socketio.emit('newImg', response.content)
    # return response


    data = response.content
    print('gen Image!!! newwww')
    if not os.path.exists('images'):
        os.makedirs('images')
    
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Split the filename into name and extension
    
    # Create a unique filename with the timestamp
    unique_filename = f"image_{timestamp}.png"
    
    image = Image.open(io.BytesIO(data))
    file_path = os.path.join('images', unique_filename)
    with open(file_path, 'wb') as file:
        image.save(file)
    
    socketio.emit("addImage", file_path)





def clean_text(text):
    """
    Cleans the input text by removing HTML tags and stopwords.
    """
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]

    # Join back into a string
    cleaned_text = ' '.join(filtered_tokens)
    
    return cleaned_text, len(filtered_tokens)

def scrape_url_content(url):
    """
    Scrapes the content of a URL and returns the cleaned text.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, and unwanted tags
        for script in soup(['script', 'style']):
            script.extract()
        
        # Get the text
        text = soup.get_text()
        
        # Clean the text
        cleaned_text, token_count = clean_text(text)
        
        return cleaned_text, token_count
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, 0
def search_web_and_answer(query):
    print("WEB QUERY: ", query)
    search_results = []
    print(f"Searching Google for '{query}'...")
    socketio.emit('new', f'''Searching Google for '{query}'...''')

    try:
        # Perform a Google search and get the top 3 links
        for result in search(query,num_results=3, lang="en"):
            search_results.append(result)

        print(f"Top 3 search results: {search_results}")
        # socketio.emit('new', f'''
        #           Top 3 search results: {search_results}
        #           ''')

        total_tokens = 0
        combined_content = ""

        # Scrape content from the top 3 links
        for url in search_results:
            if 'wikipedia.org' in url:
                print(f"Scraping Wikipedia page: {url}")
                socketio.emit('new', f'''Scraping Wikipedia page: {url}''')

                content, token_count = scrape_url_content(url)
                combined_content += content
                total_tokens += token_count
                break  # Prioritize Wikipedia link
            else:
                print(f"Scraping: {url}")
                socketio.emit('new', f'''Scraping: {url}''')

                content, token_count = scrape_url_content(url)
                combined_content += content
                total_tokens += token_count

        if not combined_content:
            return "No content found in search results."

        # Pass the combined and cleaned content to the LLM
        llm_query = f"Answer this query '{query}' using the following information: {combined_content}"
        print(f"Total token count of cleaned content: {total_tokens}")
        return ask_llama(llm_query)

    except Exception as e:
        print(e)
        return f"Error during Google search: {e}"
    
  
def get_system_info(info_type="all"):
    # Initialize variables
    system_info = ""

    # Get CPU usage
    if info_type in ["cpu", "all"]:
        cpu_usage = psutil.cpu_percent(interval=1)
        system_info += f"CPU usage is at {cpu_usage}%. "

    # Get Memory usage
    if info_type in ["memory", "all"]:
        memory = psutil.virtual_memory()
        system_info += f"Memory usage is {memory.percent}%. "

    # Get Disk usage
    if info_type in ["disk", "all"]:
        disk = psutil.disk_usage('/')
        system_info += f"Disk space is {disk.percent}% used. "

    # Optional: Add GPU usage if needed
    if info_type in ["disk", "all"]:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_load = gpu.load * 100  # Load is a value between 0 and 1, so multiply by 100
                system_info += f"GPU {gpu.name} usage is at {gpu_load}%. "
        except ImportError:
            system_info += "GPUtil not installed, GPU information is unavailable."

    if not system_info:
        system_info = "No valid parameter provided."

    socketio.emit('new', f'''{system_info}''')

    return system_info

def send_email(to_address, subject, body):
    print('EMAIL', to_address, subject, body)
    from_address = "everydaynews44@gmail.com"  # Your email
    password = "zdob qiyt uxaa rtjh"  # Your email password

    # Create the message
    msg = MIMEMultipart()
    
    # Set 'From' field to include the display name
    msg['From'] = f"NAiRa <{from_address}>"
    msg['To'] = to_address
    msg['Subject'] = subject

    # Attach the body of the email
    # body = ask_llama("Answer the user query compeltely and give only the response which is asked by the use and no other information"+body)
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_address, password)
        text = msg.as_string()
        server.sendmail(from_address, to_address, text)
        server.quit()

        print(f"Email sent to {to_address}")
        socketio.emit('new', f'''Mail Sent ......''')

        return "Sending Mail ......"
    except Exception as e:
        print(f"Failed to send email: {e}")
        return "Error sending mail ......"


def google_search(query):
    socketio.emit('new', f'''Searching Google ......''')

    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    webbrowser.open(search_url)

    return "Searching Google ......"
def youtube_search(query):
    socketio.emit('new', f'''Searching Youtube ......''')

    search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    webbrowser.open(search_url)
    open_first_youtube_link(query)


    return "Searching Youtube ......"

def open_first_youtube_link(query):
    # Perform the search using the VideosSearch class
    videos_search = VideosSearch(query, limit=1)
    search_results = videos_search.result()

    # Check if any results were returned
    if search_results and 'result' in search_results:
        first_video = search_results['result'][0]
        video_link = first_video['link']
        
        # Open the video in the web browser
        webbrowser.open(video_link)
        
        # Return the original query string
        # socketio.emit('new', f'''Opening Youtube Video......''')
        return query
    else:
        print("No video found for the query.")
    
    # If no video was found, return None

import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def ask_llama(query, rag=None):
    if rag is not None:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(F'''
Based on following context and previous questions answer the user querry:
${rag}

PREVIOUS CONVERSATIONS:
${CONTEXT}

USER QUERY:
${query}
    ''')
        print(response.text)
        socketio.emit('new', response.text)

        return response.text
    chat_completion = client.chat.completions.create(
    messages=CONTEXT,
        model="llama-3.1-8b-instant",
    )

    res = chat_completion.choices[0].message.content
    ANSWERS.append(res)
    socketio.emit('new', res)

    socketio.emit('new', f'''{res}''')
    return res

def execute_code(code_string):
    # Create a custom environment (an empty dictionary)
    # custom_env = {'client': client, 'generateImage':generateImage,'ask_llama':ask_llama,'google_search':google_search, 'youtube_search':youtube_search, 'open_first_youtube_link': open_first_youtube_link,"send_email":send_email,"get_system_info":get_system_info,"search_web_and_answer":search_web_and_answer}
    loc = {}
    try:
        # Execute the provided code string in the custom environment
        exec(code_string)
        # print(custom_env.get('result'))
        return loc
    except Exception as e:
        print(f"Error during code execution: {e}")
        return None
    



example = '''{
  {
  "ROOT_NAME": {
    "file1.extension": {},
    "file2.extension": {},
    "folder1": {
      "file3.extension": {}
    },
    "folder2": {
      "file3.extension": {},
      "file4.extension": {}
    }
  }
}
}'''

def getStructure(user_query):
    model_name="llama-3.1-8b-instant"
    temperature=0.5
    top_p=1
    # Initialize the Groq client
    client = Groq()

    # Send the user query to the LLM
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f'''
for the user query return only file structure for the code as a JSON,

query: {user_query}

stick to the following format for the json in terms of how trees is formed and not the file/folder names:
{example}

note: the leaf nodes will only contains files rest would be folders, and the json only contains files and folders, NO CODE
'''}
        ],
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        stop=None,
        stream=False,
    )

    # Extract the response content
    response = chat_completion.choices[0].message.content
    response = response.split('```json')[1].split('```')[0]
    print(response)
    # with open('ouput.txt', 'w') as f:
    #   f.write(response)
    return response



def create_file_structure(structure, user_query,base_path='/Users/pypranav/Coding/React Projects/Naira/codePlayground/'):
    for name, content in structure.items():
        # If content is a dictionary, it's a directory
        if isinstance(content, dict) and content!={}:
            dir_path = os.path.join( base_path,name)
            os.makedirs(dir_path, exist_ok=True)
            create_file_structure(content,user_query, dir_path)  # Recursively create sub-directories/files
        else:
            # If content is not a dictionary, it's a file
            file_path = os.path.join(base_path,name)
            with open(file_path, 'w') as file:
                if name.endswith(('.png', '.jpg', '.ico', '.wav')):
                    continue

                model_name="llama-3.1-8b-instant"
                temperature=0.5
                top_p=1
                # Initialize the Groq client
                client = Groq()

                # Send the user query to the LLM
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": f'''
            I want you to return code for the file I mention for the following query and file structure

            query: {user_query}
            file structure: {structure}

            return only the content to be placed inside {name} as single code file and no other text
            '''}
                    ],
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    stop=None,
                    stream=False,
                )

                # Extract the response content
                response = chat_completion.choices[0].message.content
                try:
                    response = '```'+response.split('```')[1]
                    
                    response = re.sub(r'```[a-zA-Z]+', '', response).replace('```', '')
                    
                    file.write(response)  # Create an empty file
                    
                    socketio.emit('new', f'Wrote file {name}')
                except Exception:
                    pass


import subprocess
def writeCode(query):
    try:
        socketio.emit('new', 'generating code.....')
        i=0
        while True:
            if i==10:
                break
            try:
                structure = getStructure(query)
                structure = json.loads(structure)
                break
            except Exception:
                i+=1
                continue
        create_file_structure(structure, query)
        socketio.emit('new', 'generatation finished!!!')
    except Exception:
        socketio.emit('new', 'Something went wrong!')



    abs_path = os.path.abspath('/Users/pypranav/Coding/React Projects/Naira/codePlayground/')
    subprocess.run(["code", abs_path])




def open_app(query):
    """
    Opens Spotlight search and launches an app using the given query.
    
    Parameters:
    query (str): The name of the application to search for and open.
    """
    # AppleScript to open Spotlight search, type the query, and open the first result
    socketio.emit('new',f'Opened {query}')
    apple_script = f'''
    tell application "System Events"
        keystroke space using {{command down}}
        delay 0.5
        keystroke "{query}"
        delay 0.5
        keystroke return
    end tell
    '''
    
    # Run the AppleScript command using osascript
    subprocess.run(["osascript", "-e", apple_script])




import time
import pyautogui
import pyperclip



def open_meet_and_send_mail(subject='Meet Link', to_address='sankalpdubedy@gmail.com'):
    # Open 'meet.new' in the default web browser
    webbrowser.open('https://meet.new')

    time.sleep(5)  # Wait for the browser to load the page

    while True:
        # Press Shift + Cmd + C to copy the URL (specific to macOS)
        pyautogui.hotkey('shift', 'command', 'c')
        time.sleep(1)  # Wait a moment for the clipboard to update

        # Get the current URL from the clipboard
        current_url = pyperclip.paste()
        print(current_url)
        if not current_url:
            continue

        # Check if the URL no longer contains 'new'
        if 'new' not in current_url and 'meet.google.com' in current_url:
            send_email(to_address,subject, f"Link to join meet: {current_url}")
            break

        # Wait a bit before trying again
        time.sleep(2)
    
    

@app.route('/images/<filename>')
def getImg(filename):
    return send_from_directory("images", filename)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5500)