import speech_recognition as sr
import os
import webbrowser
import time
import numpy as np
from faster_whisper import WhisperModel
from nlp_engine import NLPEngine
import re

# --- PART 1: ACTION FUNCTIONS (Unchanged) ---
def open_notepad():
    print("ACTION: Opening Notepad...")
    os.startfile('notepad.exe')

def open_website(url):
    print(f"ACTION: Opening website: {url}")
    webbrowser.open(url)

def shutdown_computer():
    print("ACTION: Initiating shutdown sequence...")
    os.system('shutdown /s /t 60')
    print("Shutdown command sent. To cancel, run 'shutdown /a' in CMD.")

def get_current_time():
    current_time = time.strftime("%I:%M %p") # e.g., "04:30 PM"
    print(f"ACTION: The current time is {current_time}")

# --- PART 2: MAIN APPLICATION WITH FASTER-WHISPER ON GPU ---

def main():
    """The main function that runs the voice assistant with Faster-Whisper on the GPU."""

    # --- SETUP FASTER-WHISPER (GPU VERSION) ---
    # !! CHANGE 1: Use a larger, more accurate model !!
    model_size = "medium.en"  # or "large-v3" for maximum accuracy

    print(f"Loading Faster-Whisper model '{model_size}'...")
    # On first run, it will download the model.
    model = WhisperModel(
        model_size,
        # !! CHANGE 2: Point to the GPU !!
        device="cuda",
        # !! CHANGE 3: Use float16 for modern GPUs !!
        compute_type="float16"
    )
    print("Model loaded successfully on GPU.")

    nlp_brain = NLPEngine()
    nlp_brain.train()

    # --- SETUP MICROPHONE (Unchanged) ---
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        microphone = sr.Microphone(sample_rate=16000)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except Exception as e:
        print(f"Could not find a working microphone: {e}")
        return

    # --- Main listening loop (Unchanged) ---
    with microphone as source:
        print("\n🎤 Voice assistant is listening... Say a command.")
        while True:
            try:
                print("Listening...")
                audio = recognizer.listen(source)
                
                print("Processing audio with Faster-Whisper on GPU...")
                start_time = time.time()
                
                raw_data = audio.get_raw_data()
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

                segments, info = model.transcribe(audio_np, beam_size=5)
                
                command = "".join(segment.text for segment in segments).strip()
                
                end_time = time.time()

                if command:
                    print(f"✅ Recognized in {end_time - start_time:.2f}s: '{command}'")
                    # process_command(command)
                    # !! NEW: USE THE NLP BRAIN TO GET THE INTENT !!
                    intent = nlp_brain.predict(command)
                    
                    # !! NEW: PROCESS THE INTENT, NOT THE RAW TEXT !!
                    process_intent(intent, command) # Pass original command for entity extraction
                else:
                    print("Could not understand audio.")

                print("\n🎤 Listening again...")

            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An unknown error occurred: {e}")
                break
                
    print("\nVoice assistant stopped.")


def process_command(command):
    # This function remains exactly the same
    processed_command = command.lower()
    if "open notepad" in processed_command:
        open_notepad()
    elif "open youtube" in processed_command:
        open_website("https://youtube.com")
    # ... etc
    elif "stop" in processed_command or "quit" in processed_command:
        raise KeyboardInterrupt
    else:
        if processed_command:
            print("❓ Command not recognized.")

def process_intent(intent, original_command):
    """Processes the recognized intent."""
    if intent == "None":
        print("❓ Not a recognized command.")
        return # Stop further processing

    if intent == "open_notepad":
        open_notepad()
    
    elif intent == "get_time":
        get_current_time()

    elif intent == "shutdown_computer":
        shutdown_computer()

    elif intent == "open_website":
        # This is our first step into "entity extraction"
        # We find the website name from the original command.
        # This is a simple version; more advanced methods exist.
        if "youtube" in original_command.lower():
            open_website("https://youtube.com")
        elif "google" in original_command.lower():
            open_website("https://google.com")
        elif "wikipedia" in original_command.lower():
            open_website("https://wikipedia.org")
        else:
            print("You asked to open a website, but I don't know which one.")

    elif intent == "greeting":
        print("Hello to you too!")

    elif intent is None:
        print("❓ I'm not sure what you mean. Please try a different command.")

    else:
        print(f"❓ Intent '{intent}' recognized, but no action is defined for it.")

if __name__ == "__main__":
    main()