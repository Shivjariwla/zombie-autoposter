import asyncio
import base64
import json
import math
import os
import random
import re
import time
import requests
import edge_tts
import cloudinary
import cloudinary.uploader
from google import genai
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips

VOICE = "en-US-AndrewNeural"  # Calm, natural, engaging storyteller voice
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]

cloudinary.config(cloudinary_url=CLOUDINARY_URL)


def generate_zombie_story():
    scenarios = [
        "locked inside a 24-hour supermarket after midnight",
        "trapped inside an IKEA warehouse after closing time",
        "stuck on the top floor of a luxury apartment building with dead power",
        "locked inside an underground subway station control booth",
        "barricaded inside a 24-hour highway gas station during a blackout",
        "trapped inside an overnight sleeper train stopped in the middle of nowhere",
        "locked inside a school building after late-night sports practice",
        "stuck inside a multi-story underground parking garage",
        "trapped inside a small-town hardware store",
        "locked inside an airport terminal after all flights are grounded",
        "stuck on a ferry boat drifting without engine power",
        "barricaded inside a movie theater projection room"
    ]

    chosen_scenario = random.choice(scenarios)

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Write a viral 40-second first-person ("you") zombie survival story about being {chosen_scenario}.
    Model the exact writing style after viral survival channels like "Nobody Gets In":

    CRITICAL WRITING RULES:
    1. SIMPLE ENGLISH ONLY: Use everyday words a 12-year-old understands. Short, punchy sentences (6 to 12 words max per sentence). Zero fancy vocabulary or poetic fluff.
    2. COLD, CALM, PRACTICAL TONE: Sound calm, smart, and matter-of-fact—not screaming or dramatic.
    3. USE REAL NUMBERS & OBJECTS: Mention exact times (e.g., "It is 2:14 in the morning"), exact counts (e.g., "40,000 items", "90 minutes of battery"), and clever real-world objects (pallet jacks, magnets, bleach, duct tape, alarm clocks).
    4. INSTANT HOOK: Sentence 1 must be a shocking, simple rule or fact that stops people from scrolling (example: "Automatic doors open for anything that walks up to them. They do not check for a pulse.").
    5. STRUCTURE: Break the story into exactly 6 short scenes (each scene = 15 to 22 words of narration). End on a tense cliffhanger problem.

    Return ONLY valid JSON:
    {{
      "caption": "POV: You're {chosen_scenario} during an outbreak. Would you survive? #zombieapocalypse #survival #pov #storytime #shorts",
      "scenes": [
        {{
          "narration": "Automatic doors open for anything that walks up to them. They do not check for a pulse.",
          "image_prompt": "First-person POV shot at night inside a brightly lit empty supermarket looking at sliding glass entrance doors, a pale motionless figure standing outside in the dark parking lot, realistic cinematic lighting, 8k, no text"
        }}
      ]
    }}
    Make sure there are exactly 6 scenes. Every image_prompt must be photorealistic first-person POV matching that exact scene, with ZERO text or words inside the image.
    """
    models_to_try = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash"]
    for attempt in range(4):
        model_name = models_to_try[attempt % len(models_to_try)]
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            clean_json = re.sub(r"```json|```", "", response.text).strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Story attempt {attempt + 1} with {model_name} failed: {e}")
            time.sleep(10)
    raise RuntimeError("All Gemini story generation attempts failed.")


def generate_ai_image(prompt, filename):
    cf_account = os.environ["CF_ACCOUNT_ID"].strip()
    cf_token = os.environ["CF_API_TOKEN"].strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {"Authorization": f"Bearer {cf_token}"}

    for attempt in range(3):
        res = requests.post(
            url,
            headers=headers,
            json={"prompt": prompt, "steps": 4},
            timeout=90
        )
        if res.status_code == 200:
            data = res.json()
            img_b64 = data["result"]["image"]
            with open(filename, "wb") as f:
                f.write(base64.b64decode(img_b64))
            return
        print(f"Attempt {attempt + 1} failed ({res.status_code}): {res.text[:200]}")
        time.sleep(5)

    raise RuntimeError("Failed to generate AI image via Cloudflare.")


async def generate_scene_audio(text, audio_path):
    communicate = edge_tts.Communicate(text, VOICE, rate="+6%")
    await communicate.save(audio_path)


def split_into_subtitle_chunks(text, total_duration):
    words = text.strip().split()
    if not words:
        return []
    # Group into 3-4 words per subtitle card so viewers can read effortlessly
    chunk_size = 4
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    total_chars = sum(len(c) for c in chunks)

    timed_chunks = []
    current_time = 0.0
    for c in chunks:
        # Weight display time by character length so longer phrases stay on screen longer
        share = (len(c) / total_chars) * total_duration if total_chars > 0 else (total_duration / len(chunks))
        dur = max(0.35, share)
        if current_time + dur > total_duration:
            dur = max(0.1, total_duration - current_time)
        timed_chunks.append((c.upper(), current_time, dur))
        current_time += dur
    return timed_chunks


def build_pov_video(story, output_path="zombie_pov.mp4"):
    scene_clips = []
    for idx, scene in enumerate(story["scenes"]):
        img_file = f"scene_{idx}.jpg"
        aud_file = f"scene_{idx}.mp3"

        generate_ai_image(scene["image_prompt"], img_file)
        asyncio.run(generate_scene_audio(scene["narration"], aud_file))

        audio = AudioFileClip(aud_file)
        dur = audio.duration

        # Smooth slow zoom-in for cinematic immersion
        base_img = (
            ImageClip(img_file)
            .with_duration(dur)
            .resized(lambda t: 1.0 + 0.07 * (t / dur))
            .with_position(("center", "center"))
        )
        bg = CompositeVideoClip([base_img], size=(1080, 1920))
        layers = [bg]

        # Guaranteed bold, readable subtitles with dark backing box
        subtitle_chunks = split_into_subtitle_chunks(scene["narration"], dur)
        for phrase, start_t, sub_dur in subtitle_chunks:
            txt = (
                TextClip(
                    text=f" {phrase} ",
                    font_size=68,
                    color="#FFE600",
                    bg_color="rgba(0,0,0,0.72)",
                    stroke_color="black",
                    stroke_width=3,
                    method="caption",
                    size=(920, None)
                )
                .with_position(("center", 1320))
                .with_start(start_t)
                .with_duration(sub_dur)
            )
            layers.append(txt)

        scene_comp = CompositeVideoClip(layers, size=(1080, 1920)).with_audio(audio)
        scene_clips.append(scene_comp)

    final_video = concatenate_videoclips(scene_clips, method="compose")
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path


def publish_to_all_platforms(public_video_url, caption):
    url = "https://api.buffer.com"
    headers = {
        "Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    org_query = """
    query {
      account {
        organizations {
          id
        }
      }
    }
    """
    org_res = requests.post(url, headers=headers, json={"query": org_query}).json()
    if "errors" in org_res:
        raise RuntimeError(f"Buffer auth/org error: {org_res['errors']}")

    orgs = org_res["data"]["account"]["organizations"]
    if not orgs:
        raise RuntimeError("No Buffer organizations found.")
    org_id = orgs[0]["id"]

    channels_query = """
    query GetChannels($orgId: OrganizationId!) {
      channels(input: { organizationId: $orgId }) {
        id
        name
        service
        isDisconnected
      }
    }
    """
    ch_res = requests.post(
        url,
        headers=headers,
        json={"query": channels_query, "variables": {"orgId": org_id}}
    ).json()

    channels = ch_res.get("data", {}).get("channels", [])
    active_channels = [c for c in channels if not c.get("isDisconnected")]
    if not active_channels:
        raise RuntimeError(f"No active channels connected in Buffer. Response: {ch_res}")

    create_post_mutation = """
    mutation CreatePost($input: CreatePostInput!) {
      createPost(input: $input) {
        ... on PostActionSuccess {
          post {
            id
            status
          }
        }
        ... on MutationError {
          message
        }
      }
    }
    """
    for ch in active_channels:
        service = ch["service"].lower()
        post_input = {
            "text": caption,
            "channelId": ch["id"],
            "schedulingType": "automatic",
            "mode": "shareNow",
            "assets": [
                {
                    "video": {
                        "url": public_video_url
                    }
                }
            ]
        }

        if service == "instagram":
            post_input["metadata"] = {
                "instagram": {
                    "type": "reel",
                    "shouldShareToFeed": True
                }
            }
        elif service == "youtube":
            yt_title = caption.split("#")[0].strip()[:90] or "POV: Zombie Outbreak Survival"
            if "#shorts" not in yt_title.lower():
                yt_title = f"{yt_title} #shorts"[:100]
            post_input["metadata"] = {
                "youtube": {
                    "title": yt_title,
                    "categoryId": "24"
                }
            }

        post_res = requests.post(
            url,
            headers=headers,
            json={"query": create_post_mutation, "variables": {"input": post_input}}
        ).json()
        print(f"Published to {ch['service']} ({ch['name']}):", post_res)


if __name__ == "__main__":
    story = generate_zombie_story()
    video_file = build_pov_video(story)

    upload = cloudinary.uploader.upload(video_file, resource_type="video")
    try:
        publish_to_all_platforms(upload["secure_url"], story["caption"])
        time.sleep(60)
    finally:
        cloudinary.uploader.destroy(upload["public_id"], resource_type="video")
