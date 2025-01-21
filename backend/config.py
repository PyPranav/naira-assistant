CORE_PROMPT = '''
    Setting: Only return one funtion

    Function 
    def search_web_and_answer(query):
      """
      To get inforamtion on any topic for answering user query
      Searches for the given query on web and then generates an intelligent response using an LLM (Large Language Model)
      by passing the web context to the LLM function (ask_llama).
      

      Args:
          query (str): The search query for which information is needed.

      Returns:
          str: The response generated by the LLM using the Wikipedia summary and query context. If no Wikipedia page is found,
              it will return a message indicating that no page was found.
      """

    Function 
    def ask_llama(query):
      """
      To generate response from an LLM, this to be used as the default when other function dont fit the need 
      NOT TO BE USED WHEN INFO FROM WEB IS ASKED
      

      Args:
          query (str): The search query for which information is needed.

      Returns:
          str: LLM response.
      """

'''


MAIL_PROMPT = '''

    Function
    def send_email(to_address, subject, body):
      """
        Only when prompt has the word MAIL in it and to be used to send mails only if user directly ask to
      Args:
      to_address (str): The recipient's email address.
      subject (str): The subject of the email.
      body (str): The body content of the email.
      """

'''

SYSTEM_INFO_PROMPT = '''

    Function
    def get_system_info():
      """
      Retrieves CPU usage, memory usage, and disk space.
      Args:
        info_type (str): The type of system information to retrieve.
                         Options:
                         - "cpu" : Retrieves CPU utilization percentage.
                         - "memory" : Retrieves memory usage percentage.
                         - "disk" : Retrieves disk usage percentage.
                         - "gpu" : Retrieves GPU utilization percentage (requires GPUtil).
                         - "all" : Retrieves all system information (CPU, memory, disk).
                         Default is "all".
      Returns:
      None: Provides system information and speaks the result.
      """

'''

YOUTUBE_PROMPT = '''
    
    Function:
    def youtube_search(query):
      """
      Use this for multiple videos
      Opens the default web browser and performs a YouTube search for the provided query, and returns the query string.

      `webbrowser` module to open the default web browser and navigate to the search results page
      on YouTube.

      Args:
      query (str): The search terms to query on YouTube. This can be any string that you would
                  typically type into YouTube's search bar.

      
      Notes:
      - The function automatically replaces spaces in the query with the '+' symbol, which is the
        appropriate URL encoding for spaces in a query string.
      - If no web browser is available or there is an issue opening the browser, the `webbrowser`
        module may raise an exception, though this is rare on most systems.
      """

'''

GOOGLE_SEARCH_PROMPT = '''

    Function:
    def google_search(query):
      """
      Use for any and only no analysis google search or search or googling operation.
      Opens the default web browser and performs a Google search for the provided query, and returns the query string.

      This function constructs a Google search URL based on the input query, then uses Python's
      `webbrowser` module to open the default web browser and navigate to the search results page.

      Args:
      query (str): The search terms to query on Google. This can be any string that you would
                  typically type into Google's search bar.

      Notes:
      - The function automatically replaces spaces in the query with the '+' symbol, which is the
        appropriate URL encoding for spaces in a query string.
      - If no web browser is available or there is an issue opening the browser, the `webbrowser`
        module may raise an exception, though this is rare on most systems.

      """

'''

IMAGE_PROMPT = '''

Function:
    def generateImage(query):
      """
      Generates an image from a user query emit the image

      Args:
      query (str): query to generate image from

      Returns:
      image string

     """

'''


OTHER_PROMPTS = {

MAIL_PROMPT:['mail','email'],
SYSTEM_INFO_PROMPT:['system','cpu', 'gpu', 'memory', 'disk', 'ram'],
YOUTUBE_PROMPT:['youtube' , 'video', 'vid', "videos"],
GOOGLE_SEARCH_PROMPT:['google', 'search', 'web'],
IMAGE_PROMPT: ['image', 'generate']

}


USER_QUERRY_PROMPT = '''

    User Query: {userQuery}

'''


def getFinalPrompt(userQuery):
    FINAL_PROMPT = CORE_PROMPT
    userQuery = userQuery.lower()

    for key in OTHER_PROMPTS:
        if any([z in userQuery for z in OTHER_PROMPTS[key]]):
            FINAL_PROMPT=key+ FINAL_PROMPT
    
    FINAL_PROMPT += USER_QUERRY_PROMPT

    print(FINAL_PROMPT)

    return FINAL_PROMPT


GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web_and_answer",
            "description": "To get information on any topic by searching the web and generating a response using an LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for which information is needed.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_llama",
            "description": "Generates a response from an LLM for general queries not related to web information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query for generating a response using LLM.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Sends an email if the prompt specifically mentions sending a mail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_address": {
                        "type": "string",
                        "description": "The recipient's email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "The subject of the email.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The body content of the email.",
                    }
                },
                "required": ["to_address", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Retrieves system information like CPU usage, memory usage, disk space, and GPU utilization or all. and if not specified it will return all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "description": "The type of system information to retrieve. Options: 'cpu', 'memory', 'disk', 'gpu'. Default is 'all'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": "Performs a YouTube search and returns the search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms to query on YouTube.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Performs a Google search for the provided query and returns the search terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms to query on Google.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generates an image based on a user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to generate an image from.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "writeCode",
            "description": "Generates code from user query and saves to drive",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to generate code",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "open_app",
        "description": "Opens Spotlight search and launches an application using the given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The name of the application to search for and open using Spotlight.",
                }
            },
            "required": ["query"]
        }
    }
    },
    {
    "type": "function",
    "function": {
        "name": "open_meet_and_send_mail",
        "description": "Opens a Google Meet using meet.new, copies the Meet link from the URL, and sends an email with the link.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The subject of the email to be sent."
                },
                "to_address": {
                    "type": "string",
                    "description": "The recipient email address to send the Meet link to.",
                }
            },
            "required": ["subject", "to_address"]
        }
    }
}
]


if __name__ == '__main__':
    print(getFinalPrompt("hey search google for pokemon and get vid on it"))




