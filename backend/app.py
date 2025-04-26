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
import sys
import subprocess
import urllib.request


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
            "content": """You are Naira, a highly capable virtual assistant with a conversational and helpful personality. Your goal is to assist users with their questions and tasks in a natural, friendly way.

Key personality traits:
- Helpful and attentive: You prioritize solving the user's problems and addressing their needs
- Conversational and direct: Speak naturally without standard greetings or unnecessary introductions
- Concise: Provide complete but efficient answers without unnecessary text
- Direct: Get to the point quickly while maintaining a friendly tone
- Personalized: Remember context from the conversation and refer to it appropriately

IMPORTANT NOTE: When answering queries, do not start with phrases like "Hello, I am Naira" or similar greetings. Instead, start directly with the relevant information in a friendly and conversational tone. Your name is already known to the user, so there's no need to constantly introduce yourself.

For factual queries, prioritize accuracy with current information. For task-based requests, focus on providing clear steps or solutions. If you're uncertain about something, acknowledge it honestly rather than making up information.

Response examples:
❌ "Hello, I am Naira. The weather in New York today is sunny."
✅ "It's sunny in New York today with a high of 75°F."

❌ "As Naira, I can tell you that the capital of France is Paris."
✅ "The capital of France is Paris."

❌ "I'm Naira, your AI assistant. To solve this equation..."
✅ "First, you'll need to isolate the variable by..."

Remember to be helpful, informative, and friendly, but avoid unnecessary introductions.
"""
        }
    ]

    for z in context:
        messages.append(z)

    messages.append(
        {
            "role": "user",
            "content": userQuery,
        })

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
            "writeCode": writeCode,
            "open_app": open_app,
            "open_meet_and_send_mail": open_meet_and_send_mail,
            "adjust_volume": adjust_volume,
            "adjust_brightness": adjust_brightness
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
                
                # Add the function call to the conversation context for better continuity
                if function_name == "search_web_and_answer":
                    messages.append({
                        "role": "assistant",
                        "content": f"I searched the web for information about '{function_args.get('query', 'your question')}' and found some relevant details."
                    })
        except Exception as e:
            print(f"Error executing function: {e}")
            socketio.emit('new', 'I encountered an issue while processing your request. Is there something else I can help you with?')

    else:
        socketio.emit('new', response_message.content)



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

    try:
        API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": "Bearer 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"}
        response = requests.post(API_URL, headers=headers, json={"inputs": query})
        
        # Check if the response contains valid image data
        if response.status_code != 200:
            error_message = f"Image generation failed with status code: {response.status_code}"
            print(error_message)
            socketio.emit('new', f'''Sorry, I couldn't generate an image right now. Error: {response.status_code}''')
            return "Image generation failed"
        
        # Get binary data from response
        data = response.content
        print('Image data received, size:', len(data), 'bytes')
        
        # Validate that the data is a proper image
        try:
            # Try to open the image data to verify it's valid
            img = Image.open(io.BytesIO(data))
            img_format = img.format
            img_size = img.size
            print(f"Valid image received: {img_format}, {img_size[0]}x{img_size[1]}")
        except Exception as img_error:
            error_message = f"Received invalid image data: {str(img_error)}"
            print(error_message)
            socketio.emit('new', error_message)
            return error_message
        
        # Ensure the images directory exists
        if not os.path.exists('images'):
            os.makedirs('images')
        
        # Generate a timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"image_{timestamp}.png"
        file_path = os.path.join('images', unique_filename)
        
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(data)
        
        print(f"Image saved to {file_path}")
        
        # Send the image path to the client
        socketio.emit("addImage", file_path)
        return file_path
        
    except Exception as e:
        error_message = f"Error generating image: {str(e)}"
        print(error_message)
        socketio.emit('new', error_message)
        return error_message





def clean_text(text):
    """
    Cleans the input text by removing HTML tags, extra whitespace, and filtering stopwords.
    Also performs special handling for weather and news content.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
    
    # Remove special characters but keep periods, question marks, etc.
    text = re.sub(r'[^\w\s\.\,\?\!\-\:\;\(\)\'\"°]', '', text)
    
    # Check if the text is likely weather or news related
    is_weather_content = any(term in text.lower() for term in ['weather', 'temperature', 'forecast', 'celsius', 'fahrenheit', 'degrees', 'precipitation', 'humidity'])
    is_news_content = any(term in text.lower() for term in ['news', 'headlines', 'according to', 'reported', 'article', 'journalist', 'press release'])
    
    # Special handling for weather content - we want to keep temperature values and units
    if is_weather_content:
        # Tokenize but don't remove as many stopwords
        tokens = word_tokenize(text)
        weather_important_words = ['in', 'at', 'on', 'for', 'with', 'by', 'to', 'from', 'of', 'is', 'will', 'be', 'was', 'the']
        stop_words = set(stopwords.words('english')) - set(weather_important_words)
        filtered_tokens = [word for word in tokens if (word.lower() not in stop_words or any(c.isdigit() for c in word) or '°' in word)]
    
    # Special handling for news content - keep more context words
    elif is_news_content:
        # Tokenize but don't remove as many stopwords
        tokens = word_tokenize(text)
        news_important_words = ['in', 'at', 'on', 'for', 'with', 'by', 'to', 'from', 'of', 'is', 'will', 'be', 'was', 'the', 'a', 'an']
        stop_words = set(stopwords.words('english')) - set(news_important_words)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words or word[0].isupper()]
    
    # Standard handling for other content
    else:
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and (word.isalnum() or word in '.,-:;?!()')]
    
    # Remove very short tokens except digits
    filtered_tokens = [token for token in filtered_tokens if len(token) > 1 or token.isdigit()]
    
    # Join back into a string
    cleaned_text = ' '.join(filtered_tokens)
    
    return cleaned_text, len(filtered_tokens)

def scrape_url_content(url):
    """
    Scrapes the content of a URL and returns the cleaned text.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Get the content type to check if it's HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            print(f"Skipping non-HTML content: {content_type}")
            return None, 0
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, nav, footer, header, ads, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript']):
            element.extract()
            
        # Special handling for known domains
        if any(domain in url for domain in ['weather.com', 'accuweather.com', 'wunderground.com']):
            # Extract weather-specific content
            return extract_weather_content(soup, url), 500
        elif any(domain in url for domain in ['news.google.com', 'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com']):
            # Extract news-specific content
            return extract_news_content(soup, url), 500
        
        # For general sites, focus on the main content
        main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(class_='content')
        
        if main_content:
            # If we found a main content container, use just that
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # Otherwise use the whole page but be more aggressive with cleaning
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        cleaned_text, token_count = clean_text(text)
        
        # If token count is too low, the extraction might have failed, try a different approach
        if token_count < 50 and not main_content:
            # Try to extract paragraphs directly
            paragraphs = soup.find_all('p')
            if paragraphs:
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                cleaned_text, token_count = clean_text(text)
        
        return cleaned_text, token_count
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, 0

def extract_weather_content(soup, url):
    """Extract relevant content from weather websites"""
    result = []
    
    # Try to find temperature information
    temp_elements = soup.find_all(['span', 'div'], class_=lambda c: c and any(t in c.lower() for t in ['temperature', 'temp', 'forecast-temp', 'current-temp', 'feels-like']))
    if temp_elements:
        for element in temp_elements:
            temp_text = element.get_text(strip=True)
            if temp_text and any(c.isdigit() for c in temp_text):
                result.append(f"Temperature: {temp_text}")
    
    # Try to find condition information
    condition_elements = soup.find_all(['span', 'div', 'p'], class_=lambda c: c and any(t in c.lower() for t in ['condition', 'weather-phrase', 'description', 'forecast-condition']))
    if condition_elements:
        for element in condition_elements:
            condition_text = element.get_text(strip=True)
            if condition_text and len(condition_text) < 100:  # Avoid long text blocks
                result.append(f"Conditions: {condition_text}")
    
    # Try to find forecast information
    forecast_elements = soup.find_all(['div', 'section'], class_=lambda c: c and any(t in c.lower() for t in ['forecast', 'hourly', 'daily', 'outlook']))
    if forecast_elements:
        for element in forecast_elements:
            forecast_text = element.get_text(strip=True)
            # Only use short forecast snippets
            if forecast_text and 20 < len(forecast_text) < 500:
                result.append(f"Forecast: {forecast_text}")
                break  # Just get one forecast section to avoid too much text
    
    # If we couldn't extract structured data, fall back to general paragraphs
    if not result:
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 20 and any(term in text.lower() for term in ['weather', 'temperature', 'forecast', 'degrees', 'celsius', 'fahrenheit']):
                result.append(text)
    
    # Join all the extracted information
    if result:
        return ' '.join(result)
    
    # Last resort: just get any text from the page
    return soup.get_text(separator=' ', strip=True)

def extract_news_content(soup, url):
    """Extract relevant content from news websites"""
    result = []
    
    # Try to find headline/title
    headline = soup.find(['h1', 'h2'], class_=lambda c: c and any(t in c.lower() for t in ['headline', 'title', 'story-title']))
    if headline:
        headline_text = headline.get_text(strip=True)
        if headline_text:
            result.append(f"Headline: {headline_text}")
    
    # Try to find article summaries
    summary = soup.find(['div', 'p'], class_=lambda c: c and any(t in c.lower() for t in ['summary', 'excerpt', 'description', 'standfirst']))
    if summary:
        summary_text = summary.get_text(strip=True)
        if summary_text:
            result.append(f"Summary: {summary_text}")
    
    # Try to find article body
    article_body = soup.find(['div', 'article'], class_=lambda c: c and any(t in c.lower() for t in ['body', 'content', 'story', 'article-body']))
    if article_body:
        # Extract paragraphs from the article body
        paragraphs = article_body.find_all('p')
        if paragraphs:
            # Get up to 5 paragraphs to keep it manageable
            for i, p in enumerate(paragraphs):
                if i >= 5:
                    break
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Skip short paragraphs
                    result.append(text)
    
    # If we couldn't extract structured data, fall back to general paragraphs
    if not result:
        paragraphs = soup.find_all('p')
        for i, p in enumerate(paragraphs):
            if i >= 5:  # Limit to 5 paragraphs
                break
            text = p.get_text(strip=True)
            if text and len(text) > 30:  # Skip very short paragraphs
                result.append(text)
    
    # Join all the extracted information
    if result:
        return ' '.join(result)
    
    # Last resort: just get any text from the page
    return soup.get_text(separator=' ', strip=True)

def search_web_and_answer(query):
    print("WEB QUERY: ", query)
    search_results = []
    print(f"Searching for '{query}'...")
    socketio.emit('new', f'''Searching for '{query}'...''')
    
    # Detect query type
    query_lower = query.lower()
    is_weather_query = any(term in query_lower for term in ['weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy', 'humidity', 'climate'])
    is_news_query = any(term in query_lower for term in ['news', 'latest', 'headlines', 'breaking', 'current events', 'today\'s news'])
    
    # Specialized search terms for different query types
    if is_weather_query:
        # Add more specificity to weather queries
        location = extract_location(query)
        if location:
            enhanced_query = f"current weather forecast {location} today"
        else:
            enhanced_query = f"current weather forecast {query}"
        print(f"Enhanced weather query: {enhanced_query}")
    elif is_news_query:
        # Add specificity to news queries
        topic = extract_news_topic(query)
        if topic:
            enhanced_query = f"latest news {topic} today"
        else:
            enhanced_query = f"latest news headlines today {query}"
        print(f"Enhanced news query: {enhanced_query}")
    else:
        enhanced_query = query
    
    try:
        # Perform a Google search and get the top 3 links
        for result in search(enhanced_query, num_results=5, lang="en"):
            search_results.append(result)

        print("Top search results:", search_results)

        total_tokens = 0
        combined_content = ""
        
        # Prioritize certain sites for specific query types
        prioritized_domains = []
        if is_weather_query:
            prioritized_domains = ['weather.com', 'accuweather.com', 'weatherchannel', 'wunderground.com', 'forecast']
        elif is_news_query:
            prioritized_domains = ['news.google.com', 'reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'nytimes.com']
        
        # First try to scrape prioritized sites if this is a special query type
        scraped_priority = False
        if prioritized_domains:
            for url in search_results:
                if any(domain in url for domain in prioritized_domains):
                    try:
                        print(f"Scraping prioritized site: {url}")
                        socketio.emit('new', f'''Scraping relevant information from: {url}''')
                        
                        content, token_count = scrape_url_content(url)
                        if content and token_count > 100:  # Ensure we got meaningful content
                            combined_content = content  # Use only this source for specialized queries
                            total_tokens = token_count
                            scraped_priority = True
                            break
                    except Exception as e:
                        print(f"Error scraping prioritized site {url}: {e}")
        
        # If no priority sites were successfully scraped, try others
        if not scraped_priority:
            # Scrape content from the links
            for url in search_results:
                try:
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
                except Exception as e:
                    print(e)

        if not combined_content:
            return "No relevant content found. Please try a different search term."

        # Pass the combined and cleaned content to the LLM with specific instructions
        if is_weather_query:
            llm_query = f"Using this information about weather: {combined_content}\n\nProvide a clear, concise weather report for the location mentioned in this query: '{query}'. Include temperature, conditions, and forecast if available. Format this as a friendly, direct response without introductions like 'Hello' or 'I am Naira'."
        elif is_news_query:
            llm_query = f"Using this news information: {combined_content}\n\nProvide the latest news about '{query}'. Summarize the most important 2-3 headlines or developments in a clear, direct way. Do not include any introductions like 'Hello' or 'I am Naira'."
        else:
            llm_query = f"Using this information: {combined_content}\n\nAnswer this query directly and helpfully without introductions: '{query}'"

        print(f"Total token count of cleaned content: {total_tokens}")
        return ask_llama(llm_query)

    except Exception as e:
        print(e)
        return f"I encountered an issue while searching for information about {query}. Is there something else I can help you with?"

def extract_location(query):
    """Extract potential location from a weather query"""
    # Remove common weather-related words to isolate location
    query = query.lower()
    weather_terms = ['weather', 'temperature', 'forecast', 'rain', 'snow', 
                     'sunny', 'cloudy', 'humidity', 'climate', 'in', 'at', 
                     'for', 'the', 'current', 'today', 'tomorrow', 'degrees']
    
    for term in weather_terms:
        query = query.replace(term, ' ')
    
    # Clean up and extract potential location
    words = [word.strip() for word in query.split() if word.strip()]
    if words:
        # If we have consecutive words, join them as they might form a location name
        location = ' '.join(words)
        return location
    return None

def extract_news_topic(query):
    """Extract potential news topic from a query"""
    query = query.lower()
    news_terms = ['news', 'latest', 'headlines', 'breaking', 'current events', 
                 'today\'s', 'recent', 'updates', 'about', 'on', 'regarding']
    
    for term in news_terms:
        query = query.replace(term, ' ')
    
    # Clean up and extract potential topic
    words = [word.strip() for word in query.split() if word.strip()]
    if words:
        topic = ' '.join(words)
        return topic
    return None

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
    try:
        if rag is not None:
            model = genai.GenerativeModel("gemini-1.5-flash")
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            response = model.generate_content(
                f'''
                You are Naira, a helpful and conversational virtual assistant. Respond to the user query based on the provided context.
                Be natural, friendly, and direct with your response.
                
                IMPORTANT: Do not begin your response with greetings like "Hello" or introduce yourself as "I am Naira."
                Answer directly without unnecessary introductions.
                
                CONTEXT FROM DOCUMENT:
                {rag}
                
                PREVIOUS CONVERSATIONS:
                {CONTEXT}
                
                USER QUERY:
                {query}
                
                Your response should:
                1. Start directly with relevant information - no greetings or self-introductions
                2. Be concise but complete
                3. Sound conversational and helpful
                4. Use first-person pronouns naturally when appropriate
                
                Examples of good responses:
                ❌ "Hello! I'm Naira. Based on the document..."
                ✅ "Based on the document, the key points are..."
                
                ❌ "As your assistant, I found that..."
                ✅ "The document indicates that..."
                ''',
                safety_settings=safety_settings
            )
            print(response.text)
            socketio.emit('new', response.text)
            return response.text
        
        # For regular (non-RAG) queries
        system_message = {
            "role": "system", 
            "content": """You are Naira, a helpful and conversational virtual assistant.
            
            IMPORTANT: Do NOT begin your responses with phrases like "Hello" or "I am Naira" or other introductory phrases.
            Answer directly without unnecessary introductions.
            
            Guidelines:
            - Start with relevant information immediately
            - Keep responses concise while being friendly
            - Use first person pronouns naturally when appropriate
            - Provide accurate and helpful information
            
            Examples of good responses:
            ❌ "Hello! I'm Naira. The capital of France is Paris."
            ✅ "The capital of France is Paris."
            
            ❌ "As your assistant, I can help you with that. First..."
            ✅ "To do that, first you'll need to..."
            """
        }
        
        # Create messages including the context for continuity
        messages = [system_message]
        
        # Add context from previous conversation if available
        if CONTEXT:
            messages.extend(CONTEXT)
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        # Get completion from the model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,  # Slightly higher temperature for more natural responses
            max_tokens=1000
        )

        res = chat_completion.choices[0].message.content
        ANSWERS.append(res)
        socketio.emit('new', res)
        return res
        
    except Exception as e:
        error_message = f"I encountered an issue while processing your request: {str(e)}"
        print(f"Error in ask_llama: {e}")
        socketio.emit('new', error_message)
        return error_message

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



def writeCode(query):
    try:
        socketio.emit('new', 'Generating code...')
        
        # Create code playground directory in current working directory if it doesn't exist
        codebase_dir = os.path.join(os.getcwd(), 'codePlayground')
        if not os.path.exists(codebase_dir):
            os.makedirs(codebase_dir)
            
        # Get structure from LLM
        i = 0
        structure = None
        while i < 10:
            try:
                structure_json = getStructure(query)
                structure = json.loads(structure_json)
                break
            except Exception as e:
                print(f"Error parsing structure (attempt {i+1}): {e}")
                i += 1
                
        if not structure:
            socketio.emit('new', 'Failed to generate code structure. Please try again with a clearer request.')
            return "Failed to generate code structure"
            
        # Create the files
        create_file_structure(structure, query, codebase_dir)
        
        socketio.emit('new', 'Code generation complete! Opening in VS Code...')
        
        # Try to open VS Code with the generated files
        try:
            # Different commands for different platforms
            if os.name == 'nt':  # Windows
                # Try different ways to open VS Code on Windows
                try:
                    # Try using the code command directly
                    subprocess.run(["code", codebase_dir], shell=True, check=False)
                except:
                    # Try using VS Code from Program Files
                    vscode_paths = [
                        os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'Microsoft VS Code', 'Code.exe'),
                        os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'Microsoft VS Code', 'Code.exe'),
                        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Microsoft VS Code', 'Code.exe')
                    ]
                    
                    for path in vscode_paths:
                        if os.path.exists(path):
                            subprocess.run([path, codebase_dir], shell=True, check=False)
                            break
                    else:
                        # If VS Code not found, just open the folder in Explorer
                        os.startfile(codebase_dir)
            
            elif sys.platform == 'darwin':  # macOS
                # Try Visual Studio Code.app first
                try:
                    subprocess.run(["open", "-a", "Visual Studio Code", codebase_dir])
                except:
                    # If that fails, try the command line 'code' if it's installed
                    try:
                        subprocess.run(["code", codebase_dir])
                    except:
                        # Last resort, just open the directory
                        subprocess.run(["open", codebase_dir])
            
            else:  # Linux
                # Try the code command
                try:
                    subprocess.run(["code", codebase_dir])
                except:
                    # If it fails, try xdg-open
                    subprocess.run(["xdg-open", codebase_dir])
        
        except Exception as e:
            print(f"Could not open VS Code: {e}")
            # Just open the directory with default file explorer as fallback
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(codebase_dir)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(["open", codebase_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", codebase_dir])
            except Exception as e:
                print(f"Could not open the directory: {e}")
            
        return "Code generated successfully"
    except Exception as e:
        error_message = f"Error generating code: {str(e)}"
        print(error_message)
        socketio.emit('new', 'Something went wrong with code generation. Please try again.')
        return error_message

def create_file_structure(structure, user_query, base_path):
    for name, content in structure.items():
        # If content is a dictionary, it's a directory
        if isinstance(content, dict) and content != {}:
            dir_path = os.path.join(base_path, name)
            os.makedirs(dir_path, exist_ok=True)
            create_file_structure(content, user_query, dir_path)  # Recursively create sub-directories/files
        else:
            # Ensure the filename has an appropriate extension
            file_path = os.path.join(base_path, name)
            file_name, file_ext = os.path.splitext(file_path)
            
            # If no extension is provided, try to determine appropriate extension based on content or parent directory
            if not file_ext:
                # Get parent directory name to help guess the extension
                parent_dir = os.path.basename(os.path.dirname(file_path))
                
                # Guess extension based on directory or common patterns
                if 'html' in parent_dir.lower() or name.lower() in ['index', 'main']:
                    file_path = file_name + '.html'
                elif 'css' in parent_dir.lower() or name.lower() in ['style', 'styles']:
                    file_path = file_name + '.css'
                elif 'js' in parent_dir.lower() or name.lower() in ['script', 'app', 'main']:
                    file_path = file_name + '.js'
                elif 'py' in parent_dir.lower() or name.lower() in ['main', 'app']:
                    file_path = file_name + '.py'
                elif 'java' in parent_dir.lower():
                    file_path = file_name + '.java'
                elif 'react' in parent_dir.lower() or 'components' in parent_dir.lower():
                    file_path = file_name + '.jsx'
                else:
                    # Default to .txt if we can't determine
                    file_path = file_name + '.txt'
            
            with open(file_path, 'w', encoding='utf-8') as file:
                if file_path.endswith(('.png', '.jpg', '.ico', '.wav')):
                    continue

                model_name = "llama-3.1-8b-instant"
                temperature = 0.5
                top_p = 1
                
                # Get the file extension for prompting the model
                _, file_ext = os.path.splitext(file_path)
                if file_ext:
                    file_ext = file_ext[1:]  # Remove the dot
                
                try:
                    # Initialize the Groq client
                    client = Groq()
                    
                    # Create a prompt that specifies the file type/extension
                    content_prompt = f'''
                    I want you to return code for a {file_ext} file based on this query and file structure.
                    
                    Query: {user_query}
                    File structure: {structure}
                    
                    File to create: {os.path.basename(file_path)}
                    
                    Return ONLY the content to be placed inside the file. No explanations, no markdown code blocks - just the raw code.
                    Make sure the code is valid, functional, and follows best practices for {file_ext} files.
                    '''
                    
                    # Send the user query to the LLM
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a code generator that produces clean, functional code files."},
                            {"role": "user", "content": content_prompt}
                        ],
                        model=model_name,
                        temperature=temperature,
                        top_p=top_p,
                        stop=None,
                        stream=False,
                    )

                    # Extract the response content
                    response = chat_completion.choices[0].message.content
                    
                    # Clean and extract code inside triple backticks if present
                    if '```' in response:
                        try:
                            # Extract code from markdown code blocks
                            code_blocks = response.split('```')
                            # Find the block that likely contains the code (usually the second block)
                            if len(code_blocks) >= 3:
                                # Remove the language identifier if present
                                code = code_blocks[1].strip()
                                if ' ' in code or '\n' in code.split('\n')[0]:
                                    # There's a language identifier on the first line
                                    code = '\n'.join(code.split('\n')[1:])
                            else:
                                code = response
                        except Exception:
                            # If extraction fails, use the full response
                            code = response
                    else:
                        code = response
                    
                    # Write the code to the file
                    file.write(code)
                    socketio.emit('new', f'Created file: {os.path.basename(file_path)}')
                except Exception as e:
                    print(f"Error generating code for {file_path}: {e}")
                    file.write(f"// Failed to generate code for this file\n// Error: {str(e)}")
                    socketio.emit('new', f'Failed to generate code for {os.path.basename(file_path)}')
                    pass




def open_app(query):
    """
    Opens an application based on the query on Windows, macOS, or Linux.
    
    Parameters:
    query (str): The name of the application to open.
    """
    try:
        app_name = query.strip()
        socketio.emit('new', f'Opening {app_name}...')
        
        if os.name == 'nt':  # Windows
            # Clean app name for Windows
            app_name_clean = app_name.lower().replace(' ', '')
            
            # Map of common apps to their executable names and potential locations
            app_map = {
                'chrome': ['chrome.exe', 'Google\\Chrome\\Application\\chrome.exe'],
                'googlechrome': ['chrome.exe', 'Google\\Chrome\\Application\\chrome.exe'],
                'firefox': ['firefox.exe', 'Mozilla Firefox\\firefox.exe'],
                'edge': ['msedge.exe', 'Microsoft\\Edge\\Application\\msedge.exe'],
                'microsoftedge': ['msedge.exe', 'Microsoft\\Edge\\Application\\msedge.exe'],
                'word': ['WINWORD.EXE', 'Microsoft Office\\root\\Office16\\WINWORD.EXE'],
                'excel': ['EXCEL.EXE', 'Microsoft Office\\root\\Office16\\EXCEL.EXE'],
                'powerpoint': ['POWERPNT.EXE', 'Microsoft Office\\root\\Office16\\POWERPNT.EXE'],
                'outlook': ['OUTLOOK.EXE', 'Microsoft Office\\root\\Office16\\OUTLOOK.EXE'],
                'notepad': ['notepad.exe'],
                'calculator': ['calc.exe'],
                'paint': ['mspaint.exe'],
                'vscode': ['Code.exe', 'Microsoft VS Code\\Code.exe'],
                'visualstudiocode': ['Code.exe', 'Microsoft VS Code\\Code.exe'],
                'cmd': ['cmd.exe'],
                'explorer': ['explorer.exe'],
                'spotify': ['Spotify.exe', 'Spotify\\Spotify.exe'],
                'discord': ['Discord.exe', 'Discord\\app-1.0.9035\\Discord.exe'],
                'slack': ['slack.exe', 'Slack\\slack.exe'],
                'teams': ['Teams.exe', 'Microsoft\\Teams\\current\\Teams.exe'],
                'zoom': ['Zoom.exe', 'Zoom\\bin\\Zoom.exe']
            }
            
            # Track if any method succeeds
            success = False
            
            # Method 1: Enhanced Start Menu search with PowerShell (most reliable)
            try:
                # PowerShell script to find and launch app via Start Menu with better fallbacks
                ps_script = f'''
                function Start-App {{
                    param([string]$AppName)
                    
                    # Try to launch directly first
                    try {{ 
                        Start-Process $AppName -ErrorAction Stop
                        return $true
                    }} catch {{
                        # If direct launch fails, try Start Menu search
                        try {{
                            # Method 1: Look for the app in the Start Menu
                            $shell = New-Object -ComObject WScript.Shell
                            $startMenu = $shell.SpecialFolders("StartMenu")
                            $programsPath = Join-Path $startMenu "Programs"
                            
                            # Search more thoroughly for app shortcuts
                            $appShortcuts = Get-ChildItem -Path $programsPath, $env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs -Recurse -Include "*$AppName*.lnk" -ErrorAction SilentlyContinue
                            
                            if ($appShortcuts) {{
                                # Launch the first matching shortcut
                                $shortcut = $appShortcuts[0].FullName
                                $shell.Run($shortcut)
                                return $true
                            }}
                            
                            # Method 2: Try Windows apps
                            $appxPackages = Get-AppxPackage | Where-Object {{ $_.Name -like "*$AppName*" -or $_.PackageFamilyName -like "*$AppName*" }}
                            
                            if ($appxPackages) {{
                                # Try to launch the app using its package family name and app ID
                                $package = $appxPackages[0]
                                $appId = (Get-AppxPackageManifest $package).Package.Applications.Application.Id
                                Start-Process "shell:AppsFolder\\$($package.PackageFamilyName)!$appId"
                                return $true
                            }}
                            
                            # Method 3: Use Start Menu search via UI automation
                            # Open Start Menu
                            $wshell = New-Object -ComObject wscript.shell
                            $wshell.SendKeys("^{ESC}")
                            Start-Sleep -Milliseconds 500
                            
                            # Type the app name
                            [System.Windows.Forms.SendKeys]::SendWait("$AppName")
                            Start-Sleep -Milliseconds 1000
                            
                            # Press Enter to launch the first result
                            [System.Windows.Forms.SendKeys]::SendWait("~")
                            
                            return $true
                        }}
                        catch {{
                            # Method 4: Complete fallback to Run dialog
                            try {{
                                $wshell = New-Object -ComObject wscript.shell
                                $wshell.Run("$AppName")
                                return $true
                            }} catch {{
                                return $false
                            }}
                        }}
                    }}
                }}
                
                # Add required type for SendKeys
                Add-Type -AssemblyName System.Windows.Forms
                
                # Try to start the app
                Start-App -AppName "{app_name}"
                '''
                
                # Save to temporary file
                ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'start_app.ps1')
                with open(ps_path, 'w') as f:
                    f.write(ps_script)
                
                # Run with bypass execution policy
                result = subprocess.run(
                    ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                    capture_output=True, 
                    text=True
                )
                
                # Clean up temporary file
                try:
                    os.remove(ps_path)
                except:
                    pass
                
                if "True" in result.stdout:
                    success = True
                
                if success:
                    return f"Opened {app_name} successfully"
            except Exception as e:
                print(f"Start Menu search method failed: {e}")
            
            # Method 2: Try the raw app name with known names
            if not success:
                for method in ['startfile', 'mapped', 'run_dialog']:
                    try:
                        if method == 'startfile':
                            # Try direct startfile
                            os.startfile(app_name)
                            success = True
                            break
                        elif method == 'mapped' and app_name_clean in app_map:
                            # Try known executable names
                            for exe in app_map[app_name_clean]:
                                try:
                                    # Try direct path
                                    os.startfile(exe)
                                    success = True
                                    break
                                except:
                                    pass
                                    
                                # Try in Program Files
                                program_files_paths = [
                                    os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), exe),
                                    os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), exe),
                                    os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\Users\\' + os.getlogin() + '\\AppData\\Local'), exe)
                                ]
                                
                                for path in program_files_paths:
                                    try:
                                        os.startfile(path)
                                        success = True
                                        break
                                    except:
                                        pass
                                        
                                if success:
                                    break
                        elif method == 'run_dialog':
                            # Last resort, use Run dialog
                            subprocess.run(f'cmd /c start "" "{app_name}"', shell=True)
                            success = True
                            break
                    except Exception as e:
                        print(f"Method {method} failed: {e}")
            
            if success:
                return f"Opened {app_name}"
            else:
                # Last resort, use UI automation to open Start Menu and type the app name
                try:
                    # Use PowerShell to automate UI interaction
                    ps_script = f'''
                    Add-Type -AssemblyName System.Windows.Forms
                    
                    # Press Windows key to open Start Menu
                    [System.Windows.Forms.SendKeys]::SendWait("^{{ESC}}")
                    Start-Sleep -Milliseconds 500
                    
                    # Type the app name
                    [System.Windows.Forms.SendKeys]::SendWait("{app_name}")
                    Start-Sleep -Milliseconds 1000
                    
                    # Press Enter to launch the first result
                    [System.Windows.Forms.SendKeys]::SendWait("~")
                    '''
                    
                    # Save script to temporary file
                    ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'start_menu_search.ps1')
                    with open(ps_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Run PowerShell script
                    subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path], 
                                  check=True, shell=True)
                    
                    # Clean up temporary file
                    try:
                        os.remove(ps_path)
                    except:
                        pass
                    
                    socketio.emit('new', f"Opening {app_name} via Start Menu search")
                    return f"Attempted to open {app_name} via Start Menu search"
                except Exception as e:
                    print(f"Start Menu UI automation failed: {e}")
                    socketio.emit('new', f"I couldn't find {app_name} automatically. Please try opening it manually.")
                    return f"Failed to open {app_name}"
            
        elif sys.platform == 'darwin':  # macOS
            try:
                # Try direct open command first (most reliable for installed apps)
                subprocess.run(["open", "-a", app_name], check=True)
                return f"Opened {app_name}"
            except:
                # Fallback to AppleScript with Spotlight search
                apple_script = f'''
                tell application "System Events"
                    -- Try direct open first
                    try
                        tell application "{app_name}" to activate
                    on error
                        -- Fallback to Spotlight
                        keystroke space using {{command down}}
                        delay 0.5
                        keystroke "{app_name}"
                        delay 0.5
                        keystroke return
                    end try
                end tell
                '''
                
                # Run the AppleScript command
                subprocess.run(["osascript", "-e", apple_script])
                return f"Opened {app_name}"
            
        else:  # Linux
            # Try multiple methods on Linux
            try:
                # First try with the default application handler
                subprocess.Popen(["xdg-open", app_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return f"Opened {app_name}"
            except:
                try:
                    # Try direct command
                    subprocess.Popen([app_name.lower()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return f"Opened {app_name}"
                except:
                    try:
                        # Try with which to find the path
                        app_path = subprocess.check_output(["which", app_name.lower()]).decode().strip()
                        subprocess.Popen([app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        return f"Opened {app_name}"
                    except:
                        try:
                            # Last resort: Try opening app launcher menu and let user select
                            if subprocess.run(["which", "dmenu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                                # If dmenu is available
                                subprocess.Popen(f"dmenu_run", shell=True)
                                socketio.emit('new', f"Opened application menu. Please select {app_name} manually.")
                                return f"Opened application menu for {app_name}"
                            elif subprocess.run(["which", "rofi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                                # If rofi is available
                                subprocess.Popen(f"rofi -show drun", shell=True)
                                socketio.emit('new', f"Opened application menu. Please select {app_name} manually.")
                                return f"Opened application menu for {app_name}"
                            else:
                                # Try standard desktop environment app menu
                                socketio.emit('new', f"Could not find {app_name}. Please try opening it manually.")
                                return f"Could not find {app_name}"
                        except:
                            socketio.emit('new', f"Could not find {app_name}. Please try opening it manually.")
                            return f"Could not find {app_name}"
            
    except Exception as e:
        error_message = f"Error opening {query}: {str(e)}"
        print(error_message)
        socketio.emit('new', f"I had trouble opening {query}. It might not be installed or accessible.")
        return error_message




import time
import pyautogui
import pyperclip



def open_meet_and_send_mail(subject='Meet Link', to_address='sankalpdubedy@gmail.com'):
    """
    Opens a new Google Meet, copies the link, and sends it via email.
    Works on both Windows and macOS.
    
    Parameters:
    subject (str): Email subject line
    to_address (str): Email recipient address
    """
    # First emit status to let user know what's happening
    socketio.emit('new', 'Creating a new Google Meet and sending the link by email...')
    
    # Open meet.new in default browser
    webbrowser.open('https://meet.new')
    
    # Wait for the page to load (adjust time as needed)
    time.sleep(5)
    
    # Initialize with empty URL
    current_url = ""
    attempts = 0
    max_attempts = 10
    
    while attempts < max_attempts:
        try:
            # Try to copy the current URL to clipboard
            if os.name == 'nt':  # Windows
                # Use Ctrl+L to focus address bar and Ctrl+C to copy
                pyautogui.hotkey('ctrl', 'l')
                time.sleep(0.5)
                pyautogui.hotkey('ctrl', 'c')
            else:  # macOS
                # Use Cmd+L to focus address bar and Cmd+C to copy
                pyautogui.hotkey('command', 'l')
                time.sleep(0.5)
                pyautogui.hotkey('command', 'c')
            
            # Short pause to ensure clipboard is updated
            time.sleep(1)
            
            # Get URL from clipboard
            current_url = pyperclip.paste()
            print(f"Copied URL: {current_url}")
            
            # Check if this is a proper Google Meet URL
            if current_url and 'meet.google.com' in current_url and 'new' not in current_url:
                # Valid meet URL found, send email
                socketio.emit('new', f'Meet created! URL: {current_url}\nSending email to {to_address}...')
                send_email(to_address, subject, f"Join me in a Google Meet: {current_url}")
                return f"Meet URL {current_url} sent to {to_address}"
        
        except Exception as e:
            print(f"Error copying URL: {e}")
        
        # Increment counter and wait before next attempt
        attempts += 1
        time.sleep(2)
    
    # If we couldn't get the URL after several attempts
    error_msg = "Couldn't capture the Google Meet URL after several attempts."
    socketio.emit('new', error_msg)
    return error_msg



@app.route('/images/<filename>')
def getImg(filename):
    return send_from_directory("images", filename)

def adjust_volume(query):
    """
    Adjusts the system volume based on the query.
    Handles requests like 'increase volume', 'decrease volume', 'volume up/down', 
    or 'set volume to X%'.
    
    Parameters:
    query (str): The volume adjustment request.
    """
    try:
        query = query.lower()
        
        # Extract specific volume level if mentioned
        specific_volume = None
        volume_match = re.search(r'(?:set|to|at)\s+(?:volume\s+)?(?:to\s+)?(\d+)(?:\s*%)?', query)
        if volume_match:
            specific_volume = int(volume_match.group(1))
            specific_volume = max(0, min(100, specific_volume))  # Ensure volume is between 0-100%
        
        # Determine if volume should increase or decrease
        increase = any(term in query for term in ['up', 'increase', 'higher', 'louder', 'raise'])
        decrease = any(term in query for term in ['down', 'decrease', 'lower', 'quieter', 'reduce'])
        mute = any(term in query for term in ['mute', 'silent', 'silence', 'off'])
        unmute = any(term in query for term in ['unmute', 'sound on', 'audio on'])
        
        # Determine the step size (default: 10%)
        step = 10
        step_match = re.search(r'by\s+(\d+)(?:\s*%)?', query)
        if step_match:
            step = int(step_match.group(1))
            step = max(1, min(100, step))  # Ensure step is reasonable
            
        # Platform-specific volume control
        if os.name == 'nt':  # Windows
            try:
                # Try multiple methods to ensure volume control works
                
                # Method 1: Using pycaw (Windows Core Audio API)
                try:
                    from comtypes import CLSCTX_ALL
                    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                    import math
                    
                    devices = AudioUtilities.GetSpeakers()
                    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    volume = interface.QueryInterface(IAudioEndpointVolume)
                    
                    # Get current volume
                    current_vol = round(volume.GetMasterVolumeLevelScalar() * 100)
                    
                    if specific_volume is not None:
                        # Set to specific volume
                        volume.SetMasterVolumeLevelScalar(specific_volume / 100, None)
                        socketio.emit('new', f"Volume set to {specific_volume}%")
                    elif increase:
                        # Increase volume
                        new_volume = min(100, current_vol + step)
                        volume.SetMasterVolumeLevelScalar(new_volume / 100, None)
                        socketio.emit('new', f"Volume increased to {new_volume}%")
                    elif decrease:
                        # Decrease volume
                        new_volume = max(0, current_vol - step)
                        volume.SetMasterVolumeLevelScalar(new_volume / 100, None)
                        socketio.emit('new', f"Volume decreased to {new_volume}%")
                    elif mute:
                        # Mute volume
                        volume.SetMute(1, None)
                        socketio.emit('new', "Volume muted")
                    elif unmute:
                        # Unmute volume
                        volume.SetMute(0, None)
                        socketio.emit('new', "Volume unmuted")
                    else:
                        socketio.emit('new', f"Current volume is {current_vol}%")
                    
                    return f"Volume control successful"
                except Exception as e:
                    print(f"pycaw method failed: {e}")
                    # Fall back to other methods
                
                # Method 2: Using PowerShell
                if specific_volume is not None:
                    # Set to specific volume
                    ps_script = f'''
                    Add-Type -TypeDefinition @'
                    using System.Runtime.InteropServices;
                    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IAudioEndpointVolume {{
                        // f(), g(), ... are unused COM method slots. Define these if you care
                        int f(); int g(); int h(); int i();
                        int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext);
                        int j();
                        int GetMasterVolumeLevelScalar(out float pfLevel);
                        int k(); int l(); int m(); int n();
                        int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, System.Guid pguidEventContext);
                        int GetMute(out bool pbMute);
                    }}
                    [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDevice {{
                        int Activate(ref System.Guid id, int clsCtx, int activationParams, out IAudioEndpointVolume aev);
                    }}
                    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDeviceEnumerator {{
                        int f(); // Unused
                        int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
                    }}
                    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject {{ }}
                    public class Audio {{
                        static IAudioEndpointVolume Vol() {{
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(/*eRender*/ 0, /*eMultimedia*/ 1, out dev));
                            IAudioEndpointVolume epv = null;
                            var epvid = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref epvid, /*CLSCTX_ALL*/ 23, 0, out epv));
                            return epv;
                        }}
                        public static float Volume {{
                            get {{
                                float v = 0;
                                Marshal.ThrowExceptionForHR(Vol().GetMasterVolumeLevelScalar(out v));
                                return v * 100;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMasterVolumeLevelScalar(value / 100, System.Guid.Empty));
                            }}
                        }}
                        public static bool Mute {{
                            get {{
                                bool mute = false;
                                Marshal.ThrowExceptionForHR(Vol().GetMute(out mute));
                                return mute;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMute(value, System.Guid.Empty));
                            }}
                        }}
                    }}
                    '@
                    
                    [Audio]::Volume = {specific_volume}
                    Write-Output "Volume set to $([Audio]::Volume)%"
                    '''
                    
                    # Save to temporary file and execute
                    ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'set_volume.ps1')
                    with open(ps_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Execute PowerShell script with bypass execution policy
                    result = subprocess.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(ps_path)
                    except:
                        pass
                    
                    socketio.emit('new', f"Volume set to {specific_volume}%")
                    return f"Volume set to {specific_volume}%"
                
                elif increase or decrease:
                    # Create volume control script
                    ps_script = f'''
                    Add-Type -TypeDefinition @'
                    using System.Runtime.InteropServices;
                    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IAudioEndpointVolume {{
                        // f(), g(), ... are unused COM method slots. Define these if you care
                        int f(); int g(); int h(); int i();
                        int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext);
                        int j();
                        int GetMasterVolumeLevelScalar(out float pfLevel);
                        int k(); int l(); int m(); int n();
                        int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, System.Guid pguidEventContext);
                        int GetMute(out bool pbMute);
                    }}
                    [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDevice {{
                        int Activate(ref System.Guid id, int clsCtx, int activationParams, out IAudioEndpointVolume aev);
                    }}
                    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDeviceEnumerator {{
                        int f(); // Unused
                        int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
                    }}
                    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject {{ }}
                    public class Audio {{
                        static IAudioEndpointVolume Vol() {{
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(/*eRender*/ 0, /*eMultimedia*/ 1, out dev));
                            IAudioEndpointVolume epv = null;
                            var epvid = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref epvid, /*CLSCTX_ALL*/ 23, 0, out epv));
                            return epv;
                        }}
                        public static float Volume {{
                            get {{
                                float v = 0;
                                Marshal.ThrowExceptionForHR(Vol().GetMasterVolumeLevelScalar(out v));
                                return v * 100;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMasterVolumeLevelScalar(value / 100, System.Guid.Empty));
                            }}
                        }}
                        public static bool Mute {{
                            get {{
                                bool mute = false;
                                Marshal.ThrowExceptionForHR(Vol().GetMute(out mute));
                                return mute;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMute(value, System.Guid.Empty));
                            }}
                        }}
                    }}
                    '@
                    
                    $currentVol = [Audio]::Volume
                    $change = {step}
                    $newVol = [Math]::{'Max(0, [Math]::Min(100, $currentVol ' + ('+ $change' if increase else '- $change') + '))'}
                    [Audio]::Volume = $newVol
                    Write-Output "Volume $('increased' if increase else 'decreased') to $([Audio]::Volume)%"
                    '''
                    
                    # Save to temporary file and execute
                    ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'change_volume.ps1')
                    with open(ps_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Execute PowerShell script with bypass execution policy
                    result = subprocess.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(ps_path)
                    except:
                        pass
                    
                    action = "increased" if increase else "decreased"
                    socketio.emit('new', f"Volume {action}. {result.stdout.strip()}")
                    return f"Volume {action}"
                
                elif mute or unmute:
                    ps_script = f'''
                    Add-Type -TypeDefinition @'
                    using System.Runtime.InteropServices;
                    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IAudioEndpointVolume {{
                        // f(), g(), ... are unused COM method slots
                        int f(); int g(); int h(); int i();
                        int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext);
                        int j();
                        int GetMasterVolumeLevelScalar(out float pfLevel);
                        int k(); int l(); int m(); int n();
                        int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, System.Guid pguidEventContext);
                        int GetMute(out bool pbMute);
                    }}
                    [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDevice {{
                        int Activate(ref System.Guid id, int clsCtx, int activationParams, out IAudioEndpointVolume aev);
                    }}
                    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDeviceEnumerator {{
                        int f(); // Unused
                        int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
                    }}
                    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject {{ }}
                    public class Audio {{
                        static IAudioEndpointVolume Vol() {{
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(/*eRender*/ 0, /*eMultimedia*/ 1, out dev));
                            IAudioEndpointVolume epv = null;
                            var epvid = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref epvid, /*CLSCTX_ALL*/ 23, 0, out epv));
                            return epv;
                        }}
                        public static float Volume {{
                            get {{
                                float v = 0;
                                Marshal.ThrowExceptionForHR(Vol().GetMasterVolumeLevelScalar(out v));
                                return v * 100;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMasterVolumeLevelScalar(value / 100, System.Guid.Empty));
                            }}
                        }}
                        public static bool Mute {{
                            get {{
                                bool mute = false;
                                Marshal.ThrowExceptionForHR(Vol().GetMute(out mute));
                                return mute;
                            }}
                            set {{
                                Marshal.ThrowExceptionForHR(Vol().SetMute(value, System.Guid.Empty));
                            }}
                        }}
                    }}
                    '@
                    
                    [Audio]::Mute = {'$true' if mute else '$false'}
                    Write-Output "Volume {'muted' if mute else 'unmuted'}"
                    '''
                    
                    # Save to temporary file and execute
                    ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'mute_volume.ps1')
                    with open(ps_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Execute PowerShell script with bypass execution policy
                    result = subprocess.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(ps_path)
                    except:
                        pass
                    
                    action = "muted" if mute else "unmuted"
                    socketio.emit('new', f"Volume {action}")
                    return f"Volume {action}"
                
                else:
                    # Get current volume
                    ps_script = '''
                    Add-Type -TypeDefinition @'
                    using System.Runtime.InteropServices;
                    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IAudioEndpointVolume {
                        // f(), g(), ... are unused COM method slots
                        int f(); int g(); int h(); int i();
                        int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext);
                        int j();
                        int GetMasterVolumeLevelScalar(out float pfLevel);
                        int k(); int l(); int m(); int n();
                        int SetMute([MarshalAs(UnmanagedType.Bool)] bool bMute, System.Guid pguidEventContext);
                        int GetMute(out bool pbMute);
                    }
                    [Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDevice {
                        int Activate(ref System.Guid id, int clsCtx, int activationParams, out IAudioEndpointVolume aev);
                    }
                    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
                    interface IMMDeviceEnumerator {
                        int f(); // Unused
                        int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
                    }
                    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumeratorComObject { }
                    public class Audio {
                        static IAudioEndpointVolume Vol() {
                            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
                            IMMDevice dev = null;
                            Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(/*eRender*/ 0, /*eMultimedia*/ 1, out dev));
                            IAudioEndpointVolume epv = null;
                            var epvid = typeof(IAudioEndpointVolume).GUID;
                            Marshal.ThrowExceptionForHR(dev.Activate(ref epvid, /*CLSCTX_ALL*/ 23, 0, out epv));
                            return epv;
                        }
                        public static float Volume {
                            get {
                                float v = 0;
                                Marshal.ThrowExceptionForHR(Vol().GetMasterVolumeLevelScalar(out v));
                                return v * 100;
                            }
                            set {
                                Marshal.ThrowExceptionForHR(Vol().SetMasterVolumeLevelScalar(value / 100, System.Guid.Empty));
                            }
                        }
                        public static bool Mute {
                            get {
                                bool mute = false;
                                Marshal.ThrowExceptionForHR(Vol().GetMute(out mute));
                                return mute;
                            }
                            set {
                                Marshal.ThrowExceptionForHR(Vol().SetMute(value, System.Guid.Empty));
                            }
                        }
                    }
                    '@
                    
                    $currentVol = [Math]::Round([Audio]::Volume)
                    $muteStatus = [Audio]::Mute
                    Write-Output "Current volume: $currentVol% ($($muteStatus ? 'Muted' : 'Unmuted'))"
                    '''
                    
                    # Save to temporary file and execute
                    ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'get_volume.ps1')
                    with open(ps_path, 'w') as f:
                        f.write(ps_script)
                    
                    # Execute PowerShell script with bypass execution policy
                    result = subprocess.run(
                        ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(ps_path)
                    except:
                        pass
                    
                    socketio.emit('new', result.stdout.strip())
                    return result.stdout.strip()
                
            except Exception as e:
                print(f"Windows volume control error: {str(e)}")
                # Use nircmd as a last resort
                try:
                    # Check if nircmd exists in the resources directory
                    nircmd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'nircmd.exe')
                    if not os.path.exists(nircmd_path):
                        # Create resources directory if it doesn't exist
                        os.makedirs(os.path.dirname(nircmd_path), exist_ok=True)
                        
                        # Download nircmd.exe for volume control
                        import urllib.request
                        urllib.request.urlretrieve('https://www.nirsoft.net/utils/nircmd.zip', 'nircmd.zip')
                        
                        # Extract nircmd.exe
                        import zipfile
                        with zipfile.ZipFile('nircmd.zip', 'r') as zip_ref:
                            zip_ref.extract('nircmd.exe', os.path.dirname(nircmd_path))
                        
                        # Clean up zip file
                        os.remove('nircmd.zip')
                    
                    if specific_volume is not None:
                        # Set to specific volume (nircmd uses 0-65535)
                        vol_value = int(65535 * (specific_volume / 100))
                        subprocess.run([nircmd_path, "setsysvolume", str(vol_value)])
                        socketio.emit('new', f"Volume set to {specific_volume}%")
                    elif increase:
                        subprocess.run([nircmd_path, "changesysvolume", str(int(6553.5))])  # ~10% increase
                        socketio.emit('new', "Volume increased")
                    elif decrease:
                        subprocess.run([nircmd_path, "changesysvolume", str(int(-6553.5))])  # ~10% decrease
                        socketio.emit('new', "Volume decreased")
                    elif mute:
                        subprocess.run([nircmd_path, "mutesysvolume", "1"])
                        socketio.emit('new', "Volume muted")
                    elif unmute:
                        subprocess.run([nircmd_path, "mutesysvolume", "0"])
                        socketio.emit('new', "Volume unmuted")
                    else:
                        # Get current volume (not supported by nircmd directly)
                        socketio.emit('new', "Volume adjustment completed")
                    
                    return "Volume adjustment completed"
                except Exception as sub_e:
                    print(f"Nircmd volume control error: {str(sub_e)}")
                    socketio.emit('new', "I couldn't adjust the volume. You may need to do it manually.")
                    return f"Error adjusting volume: {str(e)}, {str(sub_e)}"
        
        elif sys.platform == 'darwin':  # macOS
            try:
                if specific_volume is not None:
                    # Set to specific volume
                    os.system(f"osascript -e 'set volume output volume {specific_volume}'")
                    socketio.emit('new', f"Volume set to {specific_volume}%")
                    return f"Volume set to {specific_volume}%"
                elif increase:
                    # Increase volume
                    # Get current volume first
                    current_vol = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"]).decode().strip()
                    try:
                        current_vol = int(current_vol)
                        new_vol = min(100, current_vol + step)
                        os.system(f"osascript -e 'set volume output volume {new_vol}'")
                        socketio.emit('new', f"Volume increased to {new_vol}%")
                        return f"Volume increased to {new_vol}%"
                    except:
                        # If we can't get current volume, just increase relatively
                        os.system(f"osascript -e 'set volume output volume (output volume of (get volume settings) + {step})'")
                        socketio.emit('new', "Volume increased")
                        return "Volume increased"
                elif decrease:
                    # Decrease volume
                    # Get current volume first
                    current_vol = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"]).decode().strip()
                    try:
                        current_vol = int(current_vol)
                        new_vol = max(0, current_vol - step)
                        os.system(f"osascript -e 'set volume output volume {new_vol}'")
                        socketio.emit('new', f"Volume decreased to {new_vol}%")
                        return f"Volume decreased to {new_vol}%"
                    except:
                        # If we can't get current volume, just decrease relatively
                        os.system(f"osascript -e 'set volume output volume (output volume of (get volume settings) - {step})'")
                        socketio.emit('new', "Volume decreased")
                        return "Volume decreased"
                elif mute:
                    os.system("osascript -e 'set volume output muted true'")
                    socketio.emit('new', "Volume muted")
                    return "Volume muted"
                elif unmute:
                    os.system("osascript -e 'set volume output muted false'")
                    socketio.emit('new', "Volume unmuted")
                    return "Volume unmuted"
                else:
                    # Get current volume
                    current_vol = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"]).decode().strip()
                    muted = subprocess.check_output(["osascript", "-e", "output muted of (get volume settings)"]).decode().strip()
                    status = f"Current volume: {current_vol}% ({'Muted' if muted == 'true' else 'Unmuted'})"
                    socketio.emit('new', status)
                    return status
            except Exception as e:
                print(f"macOS volume control error: {str(e)}")
                socketio.emit('new', "I couldn't adjust the volume. You may need to do it manually.")
                return f"Error adjusting volume: {str(e)}"
        
        else:  # Linux
            try:
                # Detect available volume control tools
                if subprocess.run(["which", "pactl"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    # PulseAudio available
                    if specific_volume is not None:
                        # Set to specific volume
                        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{specific_volume}%"])
                        socketio.emit('new', f"Volume set to {specific_volume}%")
                        return f"Volume set to {specific_volume}%"
                    elif increase:
                        # Increase volume
                        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}%"])
                        socketio.emit('new', "Volume increased")
                        return "Volume increased"
                    elif decrease:
                        # Decrease volume
                        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}%"])
                        socketio.emit('new', "Volume decreased")
                        return "Volume decreased"
                    elif mute:
                        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
                        socketio.emit('new', "Volume muted")
                        return "Volume muted"
                    elif unmute:
                        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
                        socketio.emit('new', "Volume unmuted")
                        return "Volume unmuted"
                    else:
                        # Get current volume (more complex in pactl, simplified here)
                        socketio.emit('new', "Volume adjustment completed")
                        return "Volume adjustment completed"
                
                elif subprocess.run(["which", "amixer"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    # ALSA available
                    if specific_volume is not None:
                        # Set to specific volume
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{specific_volume}%"])
                        socketio.emit('new', f"Volume set to {specific_volume}%")
                        return f"Volume set to {specific_volume}%"
                    elif increase:
                        # Increase volume
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{step}%+"])
                        socketio.emit('new', "Volume increased")
                        return "Volume increased"
                    elif decrease:
                        # Decrease volume
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{step}%-"])
                        socketio.emit('new', "Volume decreased")
                        return "Volume decreased"
                    elif mute:
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "mute"])
                        socketio.emit('new', "Volume muted")
                        return "Volume muted"
                    elif unmute:
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "unmute"])
                        socketio.emit('new', "Volume unmuted")
                        return "Volume unmuted"
                    else:
                        # Get current volume
                        result = subprocess.run(["amixer", "-D", "pulse", "sget", "Master"], capture_output=True, text=True)
                        socketio.emit('new', "Volume adjustment completed")
                        return "Volume adjustment completed"
                
                else:
                    # Try xdotool as last resort for desktop environments
                    if subprocess.run(["which", "xdotool"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                        if increase:
                            # Simulate volume up key
                            subprocess.run(["xdotool", "key", "XF86AudioRaiseVolume"])
                            socketio.emit('new', "Volume increased")
                            return "Volume increased"
                        elif decrease:
                            # Simulate volume down key
                            subprocess.run(["xdotool", "key", "XF86AudioLowerVolume"])
                            socketio.emit('new', "Volume decreased")
                            return "Volume decreased"
                        elif mute:
                            # Simulate mute key
                            subprocess.run(["xdotool", "key", "XF86AudioMute"])
                            socketio.emit('new', "Volume muted")
                            return "Volume muted"
                        else:
                            socketio.emit('new', "Volume control not fully supported on this Linux system")
                            return "Volume control not fully supported"
                    else:
                        socketio.emit('new', "Volume control not supported on this Linux system")
                        return "Volume control not supported"
            except Exception as e:
                print(f"Linux volume control error: {str(e)}")
                socketio.emit('new', "I couldn't adjust the volume. You may need to do it manually.")
                return f"Error adjusting volume: {str(e)}"
        
        # Fallback for unsupported platforms
        socketio.emit('new', "Volume control not supported on this platform")
        return "Volume control not supported on this platform"
    
    except Exception as e:
        error_message = f"Error adjusting volume: {str(e)}"
        print(error_message)
        socketio.emit('new', "I had trouble adjusting the volume. You may need to do it manually.")
        return error_message

def adjust_brightness(percentage=None):
    """
    Adjusts the screen brightness to the specified percentage.
    
    Parameters:
    percentage (int): Brightness level from 0-100
    
    Returns:
    str: Success or error message
    """
    try:
        if percentage is None:
            socketio.emit('new', "Please specify a brightness percentage between 0 and 100.")
            return "No percentage specified"
            
        # Convert percentage to an integer and validate
        try:
            brightness_level = int(percentage)
            if brightness_level < 0:
                brightness_level = 0
            elif brightness_level > 100:
                brightness_level = 100
        except ValueError:
            socketio.emit('new', f"Invalid brightness percentage: {percentage}. Please use a number between 0 and 100.")
            return f"Invalid percentage: {percentage}"
            
        # Adjust brightness based on platform
        if os.name == 'nt':  # Windows
            try:
                # Use WMI interface to adjust brightness
                ps_script = f"""
                $brightness = {brightness_level}
                
                try {{
                    # Get WmiMonitorBrightnessMethods namespace
                    $monitors = Get-WmiObject -Namespace root\\wmi -Class WmiMonitorBrightnessMethods
                    
                    if ($monitors -ne $null) {{
                        # Set brightness for each monitor
                        foreach ($monitor in $monitors) {{
                            $monitor.WmiSetBrightness(0, $brightness)
                        }}
                        "Brightness set to $brightness%"
                    }} else {{
                        # Fallback method using PowerShell cmdlet for Windows 10+
                        (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, $brightness)
                        "Brightness set to $brightness% (using alternative method)"
                    }}
                }} catch {{
                    # Additional fallback for modern Windows versions
                    try {{
                        Add-Type -TypeDefinition @"
                        using System;
                        using System.Runtime.InteropServices;
                        
                        public class BrightnessControl {{
                            [DllImport("user32.dll")]
                            public static extern IntPtr GetDC(IntPtr hWnd);
                            
                            [DllImport("gdi32.dll")]
                            public static extern bool GetDeviceGammaRamp(IntPtr hDC, ref RAMP lpRamp);
                            
                            [DllImport("gdi32.dll")]
                            public static extern bool SetDeviceGammaRamp(IntPtr hDC, ref RAMP lpRamp);
                            
                            [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
                            public struct RAMP {{
                                [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
                                public UInt16[] Red;
                                [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
                                public UInt16[] Green;
                                [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
                                public UInt16[] Blue;
                            }}
                            
                            public static bool SetBrightness(int brightness) {{
                                IntPtr hdc = GetDC(IntPtr.Zero);
                                RAMP ramp = new RAMP();
                                ramp.Red = new ushort[256];
                                ramp.Green = new ushort[256];
                                ramp.Blue = new ushort[256];
                                
                                // Fill the gamma ramp arrays
                                for (int i = 0; i < 256; i++) {{
                                    int value = i * (brightness + 128) / 255;
                                    if (value > 255) value = 255;
                                    ushort val = (ushort)(value * 256);
                                    ramp.Red[i] = val;
                                    ramp.Green[i] = val;
                                    ramp.Blue[i] = val;
                                }}
                                
                                bool success = SetDeviceGammaRamp(hdc, ref ramp);
                                return success;
                            }}
                        }}
"@
                        $success = [BrightnessControl]::SetBrightness($brightness)
                        if ($success) {{
                            "Brightness set to $brightness% (using gamma ramp fallback)"
                        }} else {{
                            "Failed to set brightness using gamma ramp method"
                        }}
                    }} catch {{
                        "Unable to adjust brightness through Windows APIs"
                    }}
                }}
                """
                
                # Save script to temporary file
                ps_path = os.path.join(os.environ.get('TEMP', os.getcwd()), 'adjust_brightness.ps1')
                with open(ps_path, 'w') as f:
                    f.write(ps_script)
                
                # Run PowerShell script with execution policy bypass
                process = subprocess.Popen(
                    ['powershell', '-ExecutionPolicy', 'Bypass', '-File', ps_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                # Clean up temporary file
                try:
                    os.remove(ps_path)
                except:
                    pass
                
                if process.returncode == 0 and stdout:
                    message = f"Brightness set to {brightness_level}%"
                    if "alternative" in stdout or "fallback" in stdout:
                        message = f"Brightness set to {brightness_level}% (using alternative method)"
                    socketio.emit('new', message)
                    return message
                else:
                    error_msg = f"Error adjusting brightness: {stderr or 'Unknown error'}"
                    socketio.emit('new', error_msg)
                    return error_msg
                
            except Exception as e:
                # Fallback to the screen_brightness_control library if available
                try:
                    import screen_brightness_control as sbc
                    sbc.set_brightness(brightness_level)
                    socketio.emit('new', f"Brightness set to {brightness_level}% (using library fallback)")
                    return f"Brightness set to {brightness_level}% (using library fallback)"
                except ImportError:
                    # Library not available, try keyboard simulation
                    try:
                        # Using PowerShell to simulate keyboard brightness keys
                        # This isn't perfect but can work on some systems
                        for _ in range(10):  # Press brightness down multiple times to get close to minimum
                            subprocess.run('powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]0x11)"', shell=True)
                            time.sleep(0.1)
                        
                        # Each press is approximately 10% brightness
                        num_presses = int(brightness_level / 10)
                        for _ in range(num_presses):
                            subprocess.run('powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]0x12)"', shell=True)
                            time.sleep(0.1)
                        
                        socketio.emit('new', f"Brightness set to approximately {brightness_level}% (using keyboard simulation)")
                        return f"Brightness set to approximately {brightness_level}% (using keyboard simulation)"
                    except Exception as e2:
                        error_msg = f"Error adjusting brightness: {str(e)}, fallback error: {str(e2)}"
                        socketio.emit('new', error_msg)
                        return error_msg
                
        elif sys.platform == 'darwin':  # macOS
            try:
                # Convert to range 0-1
                brightness_decimal = brightness_level / 100.0
                cmd = f'osascript -e "tell application \\"System Events\\" to set display brightness to {brightness_decimal}"'
                subprocess.run(cmd, shell=True)
                socketio.emit('new', f"Brightness set to {brightness_level}%")
                return f"Brightness set to {brightness_level}%"
            except Exception as e:
                socketio.emit('new', f"Error adjusting brightness: {str(e)}")
                return f"Error: {str(e)}"
                
        else:  # Linux
            try:
                # Try xrandr for Linux (works for many desktops)
                # Get the primary display
                output = subprocess.check_output("xrandr | grep ' connected primary'", shell=True).decode()
                primary_display = output.split()[0]
                
                # Set brightness
                brightness_decimal = brightness_level / 100.0
                cmd = f"xrandr --output {primary_display} --brightness {brightness_decimal}"
                subprocess.run(cmd, shell=True)
                    
                socketio.emit('new', f"Brightness set to {brightness_level}%")
                return f"Brightness set to {brightness_level}%"
            except Exception as e:
                # Try with xbacklight as fallback
                try:
                    subprocess.run(f"xbacklight -set {brightness_level}", shell=True, check=True)
                    socketio.emit('new', f"Brightness set to {brightness_level}% (using xbacklight)")
                    return f"Brightness set to {brightness_level}% (using xbacklight)"
                except Exception as e2:
                    error_msg = f"Error adjusting brightness: {str(e)}, fallback error: {str(e2)}"
                    socketio.emit('new', error_msg)
                    return f"Error: {str(e)}"
    except Exception as e:
        error_msg = f"Failed to adjust brightness: {str(e)}"
        socketio.emit('new', error_msg)
        return error_msg


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5500)