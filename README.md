# Voice Command Assistant (Faster-Whisper + NLP)

This project is a simple voice assistant for Windows that uses speech recognition, the Faster-Whisper model (running on GPU), and a custom NLP engine to recognize and execute spoken commands. It can open Notepad, websites, tell the time, shut down the computer, and more.

## Features
- **Speech-to-Text**: Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for fast, accurate transcription on GPU.
- **NLP Intent Recognition**: Classifies user commands using a scikit-learn SVM model trained on custom data.
- **Action Execution**: Opens Notepad, websites (YouTube, Google, Wikipedia, etc.), tells the time, and can shut down the computer.
- **Extensible**: Add new intents and actions easily by editing `training_data.py` and `phase1_core_loop.py`.

## Requirements
- Windows OS (uses `os.startfile` and `notepad.exe`)
- Python 3.8+
- NVIDIA GPU with CUDA support (for Faster-Whisper GPU acceleration)

## Installation
1. Clone this repository or copy the files to your machine.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Download the desired Faster-Whisper model in advance for faster startup.

## Usage
1. Make sure your microphone is connected and working.
2. Run the main script:
   ```
   python phase1_core_loop.py
   ```
3. Speak a command (e.g., "open notepad", "what time is it", "open youtube").
4. The assistant will recognize your intent and perform the action.

## Files
- `phase1_core_loop.py`: Main application loop, microphone handling, and action execution.
- `nlp_engine.py`: NLP engine for intent recognition using scikit-learn.
- `training_data.py`: Training data for the NLP model (expandable).

## Adding New Commands
- Add new phrases/intents to `training_data.py`.
- Add new actions to `phase1_core_loop.py` in the `process_intent` function.

## Notes
- The first run may take longer as the Faster-Whisper model downloads.
- For best results, use a good quality microphone and speak clearly.
- To cancel a shutdown, run `shutdown /a` in CMD within 60 seconds.

## License
This project is for educational and personal use.
