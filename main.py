import asyncio
import base64
import json
import os
import random
import re
import time
import numpy as np
import requests
import edge_tts
from gtts import gTTS
import cloudinary
import cloudinary.uploader
from google import genai
from moviepy import (
    ImageClip,
    AudioFileClip,
    AudioArrayClip,
    CompositeAudioClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut

VOICES_TO_TRY = [
    "en-US-AndrewNeural",
    "en-US-ChristopherNeural",
    "en-US-BrianNeural",
    "en-GB-RyanNeural"
]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]
STATE_FILE = "story_state.json"

cloudinary.config(cloudinary_url=CLOUDINARY_URL)


def load_story_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("pending_part_2"):
                    return data
        except Exception as e:
            print(f"Could not read {STATE_FILE}: {e}")
    return None


def save_story_state(state_data):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)


def generate_zombie_story():
    saved_state = load_story_state()
    client = genai.Client(api_key=GEMINI_API_KEY)

    if saved_state:
        scenario = saved_state["scenario"]
        part1_summary = saved_state["part1_script"]
        last_line = saved_state["last_line"]
        mode = "PART_2"

        prompt = f"""
        Write PART 2 (the dramatic conclusion) of this viral first-person ("you") zombie survival story set in {scenario}.
        
        HERE IS WHAT HAPPENED IN PART 1:
        {part1_summary}
        
        PART 1 ENDED ON THIS EXACT CLIFFHANGER LINE:
        "{last_line}"

        CRITICAL WRITING RULES FOR PART 2:
        1. PICK UP IMMEDIATELY: Sentence 1 of Scene 1 must continue the exact second after "{last_line}".
        2. SIMPLE ENGLISH ONLY: Everyday words a 12-year-old understands. Short, punchy sentences (6 to 12 words max).
        3. COLD, CALM, PRACTICAL TONE: Sound calm, smart, and matter-of-fact—like "Nobody Gets In".
        4. USE REAL NUMBERS & OBJECTS: Clever real-world survival math and everyday items used as traps or tools.
        5. STRUCTURE (EXACTLY 6 SCENES, 15-22 WORDS EACH):
           - Scenes 1 to 4: How you narrowly survive the cliffhanger from Part 1 using clever thinking.
           - Scenes 5 to 6: A chilling final twist or Day-30 survival inventory, ending on a strong closing thought.

        VISUAL PROMPT RULES:
        - Every image_prompt MUST be vertical 9:16 cinematic dark horror photography.
        - NEVER include human hands, arms, or fingers in the foreground. Focus on moody hallways, barricaded doors, shadows through glass, emergency red lights, flashlights on the floor, and silhouettes.

        Return ONLY valid JSON:
        {{
          "caption": "Part 2 (Final) | POV: You're {scenario} during an outbreak. Did you survive? #zombieapocalypse #survival #pov #part2 #shorts",
          "scenes": [
            {{
              "narration": "Scene 1 continuation line...",
              "image_prompt": "Vertical 9:16 dark cinematic movie still..."
            }}
          ]
        }}
        Make sure there are exactly 6 scenes and ZERO text in any image.
        """
    else:
        scenarios = [
            "locked inside a 24-hour supermarket after midnight",
            "trapped inside an IKEA warehouse after closing time",
            "stuck on the 48th floor of a luxury apartment building with dead power",
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
        scenario = random.choice(scenarios)
        mode = "PART_1"

        prompt = f"""
        Write PART 1 of a viral 40-second first-person ("you") zombie survival story about being {scenario}.
        Model the exact writing style after "Nobody Gets In":

        CRITICAL WRITING RULES:
        1. SIMPLE ENGLISH ONLY: Everyday words a 12-year-old understands. Short, punchy sentences (6 to 12 words max).
        2. COLD, CALM, PRACTICAL TONE: Sound calm, smart, and matter-of-fact—never screaming.
        3. USE REAL NUMBERS & OBJECTS: Exact times ("2:14 in the morning"), exact counts ("9,000 cans", "90 minutes"), and clever objects.
        4. INSTANT HOOK: Scene 1 must open with a shocking, simple rule or fact that stops scrolling.
        5. STRUCTURE (EXACTLY 6 SCENES, 15-22 WORDS EACH):
           - Scenes 1 to 4: Establish the building rules, supplies, and first barricade.
           - Scene 5: Something goes terribly wrong (power surges back on, a lock snaps, or living raiders pull up outside).
           - Scene 6 (HARD MID-ACTION CLIFFHANGER): MUST cut off right in the middle of an immediate life-or-death emergency—DO NOT finish the action or resolve it! The final sentence of Scene 6 MUST end with a sudden disaster followed by: "Comment Part 2 if you want to see how you survive."

        VISUAL PROMPT RULES:
        - Every image_prompt MUST be vertical 9:16 cinematic dark horror photography.
        - NEVER include human hands, arms, or fingers in the foreground.

        Return ONLY valid JSON:
        {{
          "caption": "Part 1 | POV: You're {scenario} during an outbreak. Comment Part 2 👇 #zombieapocalypse #survival #pov #storytime #shorts",
          "scenes": [
            {{
              "narration": "Automatic doors open for anything that walks up to them. They do not check for a pulse.",
              "image_prompt": "Vertical 9:16 dark cinematic movie still inside an empty modern supermarket at 2 AM, sliding glass entrance doors partially open, eerie silhouette standing motionless in the foggy dark parking lot outside, cold blue fluorescent lighting, 35mm film grain, photorealistic, no text"
            }}
          ]
        }}
        Make sure there are exactly 6 scenes and ZERO text in any image.
        """

    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash"]
    for attempt in range(4):
        model_name = models_to_try[attempt % len(models_to_try)]
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            clean_json = re.sub(r"```json|```", "", response.text).strip()
            story = json.loads(clean_json)
            story["_mode"] = mode
            story["_scenario"] = scenario
            return story
        except Exception as e:
            print(f"Story attempt {attempt + 1} with {model_name} failed: {e}")
            time.sleep(10)

    # Automatic Cloudflare Llama 3.3 70B fallback when Gemini has 503 high demand
    print("Gemini busy (503) — switching to Cloudflare Llama 3.3 70B fallback...")
    cf_account = os.environ["CF_ACCOUNT_ID"].strip()
    cf_token = os.environ["CF_API_TOKEN"].strip()
    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    cf_headers = {"Authorization": f"Bearer {cf_token}"}

    for attempt in range(3):
        try:
            res = requests.post(
                cf_url,
                headers=cf_headers,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a viral short-form survival storyteller. Output ONLY valid raw JSON with no markdown or extra commentary."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1200
                },
                timeout=60
            )
            if res.status_code == 200:
                raw_text = res.json()["result"]["response"]
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                clean_json = json_match.group(0) if json_match else raw_text
                story = json.loads(clean_json)
                story["_mode"] = mode
                story["_scenario"] = scenario
                return story
            print(f"Cloudflare story attempt {attempt + 1} failed ({res.status_code}): {res.text[:200]}")
        except Exception as e:
            print(f"Cloudflare story attempt {attempt + 1} error: {e}")
        time.sleep(5)

    raise RuntimeError("All story generation attempts failed.")


def generate_ai_image(prompt, filename):
    cf_account = os.environ["CF_ACCOUNT_ID"].strip()
    cf_token = os.environ["CF_API_TOKEN"].strip()
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {"Authorization": f"Bearer {cf_token}"}

    payload = {
        "prompt": prompt,
        "steps": 4
    }

    for attempt in range(4):
        res = requests.post(url, headers=headers, json=payload, timeout=90)
        if res.status_code == 200:
            data = res.json()
            img_b64 = data["result"]["image"]
            with open(filename, "wb") as f:
                f.write(base64.b64decode(img_b64))
            return
        print(f"Image attempt {attempt + 1} failed ({res.status_code}): {res.text[:200]}")
        time.sleep(5)

    raise RuntimeError("Failed to generate AI image via Cloudflare.")


async def _try_edge_tts(text, voice, audio_path):
    communicate = edge_tts.Communicate(text, voice, rate="+6%")
    await communicate.save(audio_path)


def generate_scene_audio(text, audio_path):
    """Tries Edge-TTS across multiple voices with retries, falling back to gTTS if blocked."""
    clean_text = re.sub(r"[^\w\s.,!?'-]", "", text).strip() or "Stay quiet. Do not move."

    for attempt in range(4):
        voice = VOICES_TO_TRY[attempt % len(VOICES_TO_TRY)]
        try:
            asyncio.run(_try_edge_tts(clean_text, voice, audio_path))
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                return
        except Exception as e:
            print(f"Edge-TTS attempt {attempt + 1} ({voice}) failed: {e}")
            time.sleep(3)

    print("Falling back to gTTS for scene audio...")
    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.save(audio_path)


def create_horror_ambience(duration, fps=44100):
    """Synthesizes a subtle dark bass drone + slow tense heartbeat pulse."""
    t = np.linspace(0, duration, int(fps * duration), endpoint=False)
    drone = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.35 * np.sin(2 * np.pi * 58.2 * t)
    beat_phase = t % 1.2
    lub = np.exp(-35 * beat_phase) * np.sin(2 * np.pi * 48 * t)
    dub = np.exp(-35 * np.maximum(0, beat_phase - 0.22)) * (beat_phase >= 0.22) * np.sin(2 * np.pi * 42 * t)
    mix = (0.045 * drone) + (0.09 * (lub + dub))
    stereo = np.column_stack((mix, mix)).astype(np.float32)
    return AudioArrayClip(stereo, fps=fps)


def split_into_subtitle_chunks(text, total_duration):
    words = text.strip().split()
    if not words:
        return []
    chunk_size = 4
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    total_chars = sum(len(c) for c in chunks)

    timed_chunks = []
    current_time = 0.0
    for c in chunks:
        share = (len(c) / total_chars) * total_duration if total_chars > 0 else (total_duration / len(chunks))
        dur = max(0.4, share)
        if current_time + dur > total_duration:
            dur = max(0.15, total_duration - current_time)
        timed_chunks.append((c, current_time, dur))
        current_time += dur
    return timed_chunks


def build_pov_video(story, output_path="zombie_pov.mp4"):
    scene_clips = []
    for idx, scene in enumerate(story["scenes"]):
        img_file = f"scene_{idx}.jpg"
        aud_file = f"scene_{idx}.mp3"

        generate_ai_image(scene["image_prompt"], img_file)
        generate_scene_audio(scene["narration"], aud_file)

        audio = AudioFileClip(aud_file)
        dur = audio.duration

        raw_clip = (
            ImageClip(img_file)
            .resized(height=1920)
            .cropped(x_center=960, width=1080)
            .with_duration(dur)
        )
        if idx % 2 == 0:
            animated_img = raw_clip.resized(lambda t, d=dur: 1.0 + 0.08 * (t / d))
        else:
            animated_img = raw_clip.resized(lambda t, d=dur: 1.08 - 0.08 * (t / d))

        bg = CompositeVideoClip([animated_img.with_position(("center", "center"))], size=(1080, 1920))
        layers = [bg]

        subtitle_chunks = split_into_subtitle_chunks(scene["narration"], dur)
        for phrase, start_t, sub_dur in subtitle_chunks:
            fade_dur = min(0.12, sub_dur * 0.25)
            txt = (
                TextClip(
                    text=f"\n{phrase}\n",
                    font_size=62,
                    color="#FFFFFF",
                    stroke_color="#0A0A0A",
                    stroke_width=2,
                    method="caption",
                    size=(900, None),
                    text_align="center"
                )
                .with_position(("center", 1240))
                .with_start(start_t)
                .with_duration(sub_dur)
                .with_effects([CrossFadeIn(fade_dur), CrossFadeOut(fade_dur)])
            )
            layers.append(txt)

        scene_comp = CompositeVideoClip(layers, size=(1080, 1920)).with_audio(audio)
        scene_clips.append(scene_comp)

    video = concatenate_videoclips(scene_clips, method="compose")
    ambience = create_horror_ambience(video.duration)
    final_audio = CompositeAudioClip([video.audio, ambience])
    final_video = video.with_audio(final_audio)

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

        if story["_mode"] == "PART_1":
            full_script = " ".join(s["narration"] for s in story["scenes"])
            last_scene_text = story["scenes"][-1]["narration"]
            save_story_state({
                "pending_part_2": True,
                "scenario": story["_scenario"],
                "part1_script": full_script,
                "last_line": last_scene_text
            })
            print("Saved Part 1 cliffhanger to story_state.json for automatic Part 2 next run.")
        else:
            save_story_state({"pending_part_2": False})
            print("Part 2 completed and cleared story_state.json.")

        time.sleep(180)
    finally:
        cloudinary.uploader.destroy(upload["public_id"], resource_type="video")
