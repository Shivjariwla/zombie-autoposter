import asyncio
import base64
import json
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

VOICE = "en-US-ChristopherNeural"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]

cloudinary.config(cloudinary_url=CLOUDINARY_URL)


def generate_zombie_story():
    locations = [
        "an empty IKEA warehouse after closing time",
        "a stalled ski-lift gondola dangling above a dark pine forest",
        "the basement laundry room of an old apartment building",
        "a flooded underground subway tunnel",
        "a bumper-to-bumper highway tunnel during a blackout",
        "the projection booth of an abandoned 1980s movie theater",
        "a remote 24-hour laundromat at 3:00 AM",
        "a walk-in freezer inside a deserted restaurant kitchen",
        "the top floor of a glass office skyscraper with dead elevators",
        "a foggy ferry boat drifting without power",
        "an overnight cross-country sleeper train stopped in the woods",
        "a dark school gymnasium with bleachers and locker rooms",
        "a crowded airport baggage claim terminal during lockdown",
        "a storm drain beneath a suburban street",
        "an isolated fire lookout tower in the mountains",
        "a pitch-black aquarium tunnel surrounded by glass tanks",
        "a multi-story underground parking garage",
        "a rural gas station convenience store during a thunderstorm"
    ]

    twists = [
        "your phone flashlight is at 3% battery and flickering",
        "a loud emergency broadcast alert suddenly blares from a nearby radio",
        "you dropped your keys behind a metal grate just feet from a horde",
        "one of the doors you thought was locked slowly creaks open on its own",
        "the zombies in this area hunt purely by sound, and broken glass covers the floor",
        "you only have one flare left and hear heavy footsteps sprinting down the hall",
        "the emergency backup generator is running out of fuel and the red lights are dying",
        "someone is banging on the outside door begging for help, but their shadow looks wrong",
        "a security camera monitor shows them already inside the building with you",
        "your walkie-talkie crackles with a warning that the building is about to be firebombed"
    ]

    chosen_location = random.choice(locations)
    chosen_twist = random.choice(twists)

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Write a terrifying, heart-pounding 30-second first-person POV zombie apocalypse horror story.
    Setting: Trapped inside {chosen_location}.
    Suspense Twist: {chosen_twist}.

    Style rules:
    - Make it genuinely scary, claustrophobic, and suspenseful (short breathless sentences, sensory details like footsteps, breathing, scratching at doors, flickering shadows).
    - End on a chilling cliffhanger or split-second survival choice.
    - Break it into exactly 5 dramatic scenes (1-2 short punchy sentences per scene).

    Return ONLY valid JSON:
    {{
      "caption": "POV: You're trapped in {chosen_location} during an outbreak... Would you survive? #zombieapocalypse #pov #scarystories #horror #survival #shorts",
      "scenes": [
        {{
          "narration": "Spoken line for scene 1...",
          "image_prompt": "First-person POV horror shot inside {chosen_location}, pitch black shadows, dim flickering emergency light, terrifying silhouettes of infected figures approaching, hyperrealistic cinematic horror, 8k, no text"
        }}
      ]
    }}
    Every image_prompt MUST be first-person POV survival horror matching that exact scene's action, dark atmospheric lighting, photorealistic, with zero text in the image.
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

        base_img = (
            ImageClip(img_file)
            .with_duration(dur)
            .resized(lambda t: 1.0 + 0.08 * (t / dur))
            .with_position(("center", "center"))
        )
        bg = CompositeVideoClip([base_img], size=(1080, 1920))
        layers = [bg]

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
