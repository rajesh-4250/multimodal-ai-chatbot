import cv2
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import google.generativeai as genai
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from io import BytesIO
import base64

# ---------- API Keys ----------
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not ELEVEN_LABS_API_KEY or not GEMINI_API_KEY:
    print("⚠️ Please set ELEVEN_LABS_API_KEY and GEMINI_API_KEY environment variables before running.")
    exit()

# Initialize ElevenLabs Client
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)


# ---------- Webcam Capture ----------
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Error: Could not open webcam.")
        return None

    print("🎥 Press 'c' to capture an image, or 'q' to quit.")

    img_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        cv2.imshow("Live Video - Press 'c' to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            img_path = "captured_frame.jpg"
            cv2.imwrite(img_path, frame)
            print(f"📸 Image saved: {img_path}")
            break
        elif key == ord("q"):
            print("❌ Exiting video capture.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return img_path


# ---------- Image Encoding ----------
def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Error encoding image: {e}")
        return None


# ---------- Record Audio ----------
def record_audio(filename="input_audio.wav", duration=5, samplerate=44100):
    print("🎙️ Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    wav.write(filename, samplerate, audio_data)
    print(f"📁 Recording saved as {filename}")
    return filename


# ---------- ElevenLabs Transcription ----------
def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.speech_to_text.convert(model_id="scribe_v1", file=audio_file)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error in transcription: {e}")
        return None


# ---------- Gemini AI Response ----------
def get_gemini_response(text, image_path=None):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={
            "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 200
        })

        # Force Gemini to reply in English
        text = f"Respond in clear English to: {text}"

        content = [{"text": text}]
        if image_path:
            image_data = encode_image(image_path)
            if image_data:
                content.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                })

        response = model.generate_content(contents=[{"parts": content}])
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error in AI response: {e}")
        return "I'm sorry, I couldn't process that."


# ---------- ElevenLabs Text-to-Speech ----------
def play_speech(text):
    try:
        response = client.text_to_speech.convert(
            model_id="eleven_multilingual_v2",
            voice_id="pNInz6obpgDQGcFmaJgB",
            output_format="pcm_16000",  # Ensures raw PCM output
            text=text,
            voice_settings=VoiceSettings(
                stability=0.5, similarity_boost=1.0, style=0.5,
                use_speaker_boost=True, speed=1.0
            ),
        )

        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        # Play audio directly
        audio_stream.seek(0)
        audio_data = np.frombuffer(audio_stream.read(), dtype=np.int16)
        sd.play(audio_data, samplerate=16000)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Error playing speech: {e}")


# ---------- Main Loop ----------
def main():
    while True:
        print("\n========== MENU ==========")
        print("1 → Talk to bot by voice")
        print("2 → Talk to bot by text")
        print("3 → Exit")
        print("==========================")

        choice = input("👉 Select an option: ").strip()

        if choice == "3":
            print("👋 Exiting...")
            break
        elif choice not in ["1", "2"]:
            print("⚠️ Invalid choice. Try again.")
            continue

        # Ask if image capture is needed
        use_image = input("📷 Do you want to capture an image? (y/n): ").strip().lower()
        image_path = capture_image() if use_image == "y" else None

        if choice == "1":
            # Voice Mode
            recorded_file = record_audio()
            audio_text = transcribe_audio(recorded_file)
            if not audio_text:
                print("⚠️ No speech detected. Try again.")
                continue
            print(f"📝 You said: {audio_text}")
            gemini_reply = get_gemini_response(audio_text, image_path)

        elif choice == "2":
            # Text Mode
            user_text = input("💬 You: ").strip()
            if not user_text:
                print("⚠️ Empty input. Try again.")
                continue
            gemini_reply = get_gemini_response(user_text, image_path)

        # Display + Speak AI Reply
        print(f"🤖 Bot: {gemini_reply}")
        play_speech(gemini_reply)


if __name__ == "__main__":
    main()