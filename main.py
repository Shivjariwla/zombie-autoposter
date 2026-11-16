import asyncio
import json
import os
import re
import time
import urllib.parse
import requests
import edge_tts
import cloudinary
import cloudinary.uploader
from google import genai
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

VOICE = "en-US-ChristopherNeural"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]

cloudinary.config(cloudinary_url=CLOUDINARY_URL)

def generate_zombie_story():
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = """
    Write a tense 30-second first-person POV zombie apocalypse survival story in a unique location (e.g. subway station, hospital corridor, gas station, apartment roof).
    Break it into 5 dramatic scenes (1-2 short punchy sentences each).
    Return ONLY valid JSON:
    {
      "caption": "POV: You're trapped during a zombie outbreak. Would you survive? #zombieapocalypse #pov #survival #horror #shorts",
      "scenes": [
        {
          "narration": "Spoken line for scene 1...",
          "image_prompt": "First-person POV shot looking through cracked glass doors at night, zombies silhouette outside in fog, flashlight beam, dark survival horror, photorealistic 8k"
        }
      ]
    }
    Include exactly 5 scenes. Every image_prompt must be first-person POV survival horror with no text in the image.
    """
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    clean_json = re.sub(r"```json|```", "", response.text).strip()
    return json.loads(clean_json)

def generate_ai_image(prompt, filename):
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&model=flux"
    for _ in range(3):
        res = requests.get(url, timeout=90)
        if res.status_code == 200 and len(res.content) > 10000:
            with open(filename, "wb") as f:
                f.write(res.content)
            return
        time.sleep(5)
    raise RuntimeError("Failed to generate AI image.")

async def generate_scene_audio(text, audio_path):
    communicate = edge_tts.Communicate(text, VOICE, rate="+12%")
    word_timings = []
    with open(audio_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 10_000_000
                duration = chunk["duration"] / 10_000_000
                word_timings.append((chunk["text"], start, start + duration))
    return word_timings

def build_pov_video(story, output_path="zombie_pov.mp4"):
    scene_clips = []
    for idx, scene in enumerate(story["scenes"]):
        img_file = f"scene_{idx}.jpg"
        aud_file = f"scene_{idx}.mp3"

        generate_ai_image(scene["image_prompt"], img_file)
        timings = asyncio.run(generate_scene_audio(scene["narration"], aud_file))

        audio = AudioFileClip(aud_file)
        dur = audio.duration

        # Dramatic first-person POV slow zoom
        base_img = (
            ImageClip(img_file)
            .with_duration(dur)
            .resized(lambda t: 1.0 + 0.08 * (t / dur))
            .with_position(("center", "center"))
        )
        bg = CompositeVideoClip([base_img], size=(1080, 1920))
        layers = [bg]

        # 3-word bold survival captions
        for i in range(0, len(timings), 3):
            chunk = timings[i:i+3]
            phrase = " ".join([w[0].upper() for w in chunk])
            start_t = chunk[0][1]
            end_t = chunk[-1][2]

            txt = (
                TextClip(
                    text=phrase,
                    font_size=76,
                    color="white",
                    stroke_color="black",
                    stroke_width=5,
                    method="caption",
                    size=(920, None)
                )
                .with_position(("center", 1350))
                .with_start(start_t)
                .with_duration(max(0.2, end_t - start_t))
            )
            layers.append(txt)

        scene_comp = CompositeVideoClip(layers, size=(1080, 1920)).with_audio(audio)
        scene_clips.append(scene_comp)

    final_video = concatenate_videoclips(scene_clips, method="compose")
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path

def publish_to_all_platforms(public_video_url, caption):
    headers = {"Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}"}
    # 1. Fetch all connected profiles (Instagram, YouTube, TikTok)
    profiles = requests.get("https://api.bufferapp.com/1/profiles.json", headers=headers).json()
    profile_ids = [p["id"] for p in profiles]

    if not profile_ids:
        raise RuntimeError("No social channels connected to Buffer.")

    # 2. Publish immediately to all 3 channels
    payload = {
        "profile_ids[]": profile_ids,
        "text": caption,
        "media[video]": public_video_url,
        "now": "true"
    }
    res = requests.post("https://api.bufferapp.com/1/updates/create.json", headers=headers, data=payload).json()
    print("Buffer publish response:", res)

if __name__ == "__main__":
    story = generate_zombie_story()
    video_file = build_pov_video(story)

    upload = cloudinary.uploader.upload(video_file, resource_type="video")
    try:
        publish_to_all_platforms(upload["secure_url"], story["caption"])
        time.sleep(60)  # Give Buffer 60s to download the video before cleanup
    finally:
        cloudinary.uploader.destroy(upload["public_id"], resource_type="video")
