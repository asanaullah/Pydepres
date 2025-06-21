# AI tools were used to clean up the code structure of a working implementation and add comments

import os
import time
import subprocess
import requests
from datetime import datetime

# --- Constants ---
# Define API endpoint and default model for the inference function
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash" # Changed to pro as it's used in the main loop
CONTAINERFILE_NAME = "Containerfile"
IMAGE_NAME = "ilab_model_serve"
IMAGE_TAG = "0.1"

# Containerfile directives that indicate a valid line
CONTAINERFILE_DIRECTIVES = ['FROM', 'WORKDIR', 'USER', 'RUN', 'COPY', 'ENV', 'LABEL', 'EXPOSE', 'ARG', 'ADD', 'VOLUME', 'STOPSIGNAL', 'HEALTHCHECK', 'ENTRYPOINT', 'CMD', 'ONBUILD', 'SHELL']

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg, init=0, end="\n"):
    global timestamp
    if init:
        with open("run_" + timestamp + ".log" ,'w') as f:
            f.write('')
    else:
        with open("run_" + timestamp + ".log",'a') as f:
            f.write(msg)
        print(msg,end=end)

# --- Helper Functions ---
def inference(context: str, prompt: str, model: str = DEFAULT_GEMINI_MODEL, max_retries: int = 10, initial_delay: int = 1) -> str:
    """
    Performs an inference call to the Gemini API with retry logic and exponential backoff.

    Args:
        context (str): The context for the prompt.
        prompt (str): The main prompt for the model.
        model (str): The Gemini model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro").
        max_retries (int): Maximum number of retries for API calls.
        initial_delay (int): Initial delay in seconds before retrying.

    Returns:
        str: The text content of the model's response.

    Raises:
        ValueError: If the GEMINI_API_KEY environment variable is not set.
        Exception: If maximum retries are exceeded or other unhandled API errors occur.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": context},
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 2048,
            "stopSequences": []
        }
    }

    for attempt in range(max_retries):
        try:
            # Use f-string for dynamic URL if model varies
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                delay = initial_delay * (2 ** attempt)
                log(f"Rate limit hit (HTTP 429). Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                # Re-raise other HTTP errors immediately as they are not typically transient
                log(f"HTTP error occurred: {e}. Status code: {e.response.status_code}")
                raise
        except requests.exceptions.ConnectionError as e:
            delay = initial_delay * (2 ** attempt)
            log(f"Connection error occurred: {e}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except requests.exceptions.Timeout as e:
            delay = initial_delay * (2 ** attempt)
            log(f"Request timed out: {e}. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            # Catch all other requests-related exceptions
            log(f"An unexpected request error occurred: {e}")
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                log(f"Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise  # Re-raise if it's the last attempt

    raise Exception(f"Max retries ({max_retries}) exceeded. Failed to get a successful response from Gemini API.")


def write_containerfile(containerfile_content: str):
    """
    Writes the provided content to the Containerfile.

    Args:
        containerfile_content (str): The content to write to the Containerfile.
    """
    log(f"--- Writing {CONTAINERFILE_NAME} ---")
    with open(CONTAINERFILE_NAME, 'w') as f:
        f.write(containerfile_content)


def build_image(image_name: str, image_tag: str) -> tuple[bool, str]:
    """
    Builds a container image using Podman from the current directory.

    Args:
        image_name (str): The name of the image.
        image_tag (str): The tag for the image.

    Returns:
        tuple[bool, str]: A tuple where the first element is True if the build was
                          successful, False otherwise, and the second element is
                          the full build log.
    """
    full_image_tag = f"{image_name}:{image_tag}"
    log(f"--- Building Image: {full_image_tag} ---")
    log_lines = []
    try:
        # Using `text=True` automatically handles decoding stdout/stderr
        env = os.environ.copy()
        env["TMPDIR"] = "/home/ahmed/tmp"
        process = subprocess.Popen(
            ["podman", "build", "-t", full_image_tag, "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env  # <- pass updated env
        )
        for line in process.stdout:
            log(line, end="")  # Print in real-time
            log_lines.append(line)
        process.stdout.close()
        return_code = process.wait()

        success = return_code == 0
        if not success:
            log(f"Image build failed with exit code {return_code}.")
    except FileNotFoundError:
        log("Error: 'podman' command not found. Please ensure Podman is installed and in your PATH.")
        success = False
    except Exception as e:
        log(f"An unexpected error occurred during image build: {e}")
        success = False

    return success, ''.join(log_lines)


def parse_containerfile_response(raw_response: str) -> str:
    """
    Parses the raw model response to extract valid Containerfile lines.
    This function attempts to filter out any non-Containerfile text the model might
    inadvertently include, relying on common Containerfile directives.

    Args:
        raw_response (str): The raw text response from the Gemini model.

    Returns:
        str: A string containing only the recognized Containerfile lines.
    """
    lines = []
    # Split by newline and process each line
    response_lines = raw_response.split('\n')

    for line in response_lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue # Skip empty lines

        # Check if the line starts with a known Containerfile directive
        # or if it's a continuation of a multi-line instruction (ends with \)
        # The split will handle cases like "RUN apt-get" vs "RUN"
        first_word = stripped_line.split(' ', 1)[0].upper() # Get the first word and make it uppercase for robust comparison

        if first_word in CONTAINERFILE_DIRECTIVES:
            lines.append(stripped_line)
        elif lines and lines[-1].endswith('\\'):
            # This line continues the previous one if the previous one ended with a backslash
            lines.append(stripped_line)
        # Optional: could add logic here for comments if desired to preserve them
        # elif stripped_line.startswith('#'):
        #     lines.append(stripped_line)

    return '\n'.join(lines)


# --- Main Logic ---
def main():
    """
    Main function to drive the Containerfile generation and image building process.
    It iteratively attempts to build an image, and if it fails, it uses Gemini
    to get an updated Containerfile based on the build logs.
    """
    log(msg="",init=1)
    initial_containerfile = """FROM fedora:latest    
RUN ilab model serve
"""
    log("#### Initial Containerfile ####")
    log(initial_containerfile)

    current_containerfile = initial_containerfile
    build_success = False
    attempt_count = 0

    while not build_success:
        attempt_count += 1
        log(f"\n#### Attempt {attempt_count}: Iterative Containerfile Generation and Build ####")

        write_containerfile(current_containerfile)
        build_success, build_log = build_image(IMAGE_NAME, IMAGE_TAG)

        if build_success:
            log("\n#### Image Built Successfully! ####")
            break
        else:
            log("\n#### Image Build Failed. Requesting Gemini for an update... ####")
            # Context for the AI model: instruct it to only output raw Containerfile code
            context = (
                "You are a helpful container building assistant for Fedora Linux that only responds with raw "
                "Containerfile code. Your only output is the Containerfile. No descriptions or details are needed. Pay extra attention to python module versions, especially when compiling modules and getting gcc or g++ errors."
            )
            # Prompt to the AI model: provide the problematic Containerfile and the build log
            prompt = (
                f"I am trying to build an image for running Instructlab using the following Containerfile:\n{current_containerfile}\n\n"
                f"Give me the updated Containerfile that resolves the following error:\n{build_log}"
            )

            try:
                raw_gemini_response = inference(context, prompt, DEFAULT_GEMINI_MODEL)
                current_containerfile = parse_containerfile_response(raw_gemini_response)
                log("\n#### Updated Containerfile from Gemini ####")
                log(current_containerfile)
            except Exception as e:
                log(f"\n--- Error during Gemini inference or response parsing: {e} ---")
                log("Exiting as AI assistance failed. Please check your API key or the error message.")
                break # Exit the loop if AI inference fails

    if not build_success:
        log("\n--- Failed to build image after multiple attempts. ---")


if __name__ == "__main__":
    main()