import asyncio
import json
import math
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
from PIL import Image, ImageDraw, ImageFont
from google import genai
from moviepy import (
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]
HISTORY_FILE = "used_topics.json"

cloudinary.config(cloudinary_url=CLOUDINARY_URL)

# Distinct neural voices for Speaker A (Boss/Pushy Person) vs Speaker B (Calm Employee/Customer)
VOICE_PAIRS = [
    ("en-US-JennyNeural", "en-US-AvaNeural"),
    ("en-US-GuyNeural", "en-US-AndrewNeural"),
    ("en-US-MichelleNeural", "en-US-BrianNeural"),
    ("en-GB-SoniaNeural", "en-GB-RyanNeural"),
]

SCENARIOS = [
    {
        "topic": "Manager expects employee to answer work emails after 5 PM without on-call pay",
        "setting": "office",
        "prop_badge": "5:02 PM",
        "role_a": "Manager",
        "role_b": "Employee",
    },
    {
        "topic": "Boss denies a PTO request submitted 3 months in advance because 'we are like a family'",
        "setting": "office",
        "prop_badge": "PTO DENIED",
        "role_a": "Boss",
        "role_b": "Worker",
    },
    {
        "topic": "HR offers a $15 gift card instead of a cost-of-living raise after record profits",
        "setting": "office",
        "prop_badge": "PIZZA PARTY",
        "role_a": "HR Rep",
        "role_b": "Employee",
    },
    {
        "topic": "Manager schedules a mandatory 'quick 5-minute sync' at 4:58 PM on a Friday",
        "setting": "office",
        "prop_badge": "FRI 4:58 PM",
        "role_a": "Manager",
        "role_b": "Employee",
    },
    {
        "topic": "Boss asks employee to train their new replacement who is getting paid more",
        "setting": "office",
        "prop_badge": "HANDOFF",
        "role_a": "Manager",
        "role_b": "Employee",
    },
    {
        "topic": "Coworker tries to take credit for a presentation in front of the director",
        "setting": "office",
        "prop_badge": "Q4 DECK",
        "role_a": "Coworker",
        "role_b": "Analyst",
    },
    {
        "topic": "Landlord tries to keep the security deposit for normal carpet wear",
        "setting": "desk",
        "prop_badge": "LEASE",
        "role_a": "Landlord",
        "role_b": "Tenant",
    },
    {
        "topic": "Customer demands a full refund on an item clearly bought two years ago",
        "setting": "counter",
        "prop_badge": "NO RECEIPT",
        "role_a": "Customer",
        "role_b": "Cashier",
    },
    {
        "topic": "Recruiter advertises 'entry-level role' that requires 7 years of experience and 4 rounds of interviews",
        "setting": "office",
        "prop_badge": "ENTRY LEVEL",
        "role_a": "Recruiter",
        "role_b": "Applicant",
    },
    {
        "topic": "Manager calls employee on their approved sick day asking them to 'just log in for an hour'",
        "setting": "office",
        "prop_badge": "SICK DAY",
        "role_a": "Manager",
        "role_b": "Employee",
    },
    {
        "topic": "Boss says 'nobody wants to work anymore' after refusing to pay market rate",
        "setting": "office",
        "prop_badge": "$14 / HR",
        "role_a": "Boss",
        "role_b": "Candidate",
    },
    {
        "topic": "Manager asks employee to cover a double shift with 10 minutes notice",
        "setting": "counter",
        "prop_badge": "URGENT SHIFT",
        "role_a": "Manager",
        "role_b": "Staff",
    },
]


def load_used_topics():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("used_topics", [])
        except Exception:
            pass
    return []


def save_used_topic(topic):
    used = load_used_topics()
    used.append(topic)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"used_topics": used[-50:]}, f, indent=2)


def pick_scenario():
    used = set(load_used_topics())
    available = [s for s in SCENARIOS if s["topic"] not in used]
    if not available:
        available = SCENARIOS
    return random.choice(available)


def generate_comedy_skit(scenario):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Write a viral 55-second relatable two-person comedy dialogue skit about:
    "{scenario['topic']}"
    Speaker A is "{scenario['role_a']}" (pushy, passive-aggressive, out of touch).
    Speaker B is "{scenario['role_b']}" (calm, polite, deadpan, sets firm boundaries with sharp logic).

    STYLE & PACING RULES:
    1. Model the comedic rhythm after "Mercy J Animation" workplace comedy Shorts.
    2. Start immediately in the middle of the conversation—no intro narration.
    3. Write 8 to 10 back-and-forth dialogue turns (alternating A and B).
    4. Each turn should be 12 to 24 words of natural, everyday spoken English.
    5. Speaker B should stay completely polite and calm while dismantling Speaker A's unreasonable demand, ending on a punchy mic-drop final line.
    6. Assign an emotion to each turn from: ["neutral", "smug", "annoyed", "shocked", "deadpan", "confident"].

    Return ONLY valid JSON:
    {{
      "caption": "POV: {scenario['role_a']}s Be Like 😂 #stickmancomedy #workhumor #pov #relatable #shorts",
      "dialogue": [
        {{
          "speaker": "A",
          "emotion": "smug",
          "text": "Good morning, do you have a quick second? I sent you an urgent email last night..."
        }},
        {{
          "speaker": "B",
          "emotion": "deadpan",
          "text": "I just clocked in right now, so I am opening my inbox as we speak."
        }}
      ]
    }}
    """

    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash"]
    for attempt in range(4):
        model_name = models_to_try[attempt % len(models_to_try)]
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            clean_json = re.sub(r"```json|```", "", response.text).strip()
            skit = json.loads(clean_json)
            if skit.get("dialogue") and len(skit["dialogue"]) >= 4:
                return skit
        except Exception as e:
            print(f"Skit attempt {attempt + 1} ({model_name}) failed: {e}")
            time.sleep(8)

    # Cloudflare Llama 3.3 70B fallback
    print("Switching to Cloudflare Llama 3.3 70B fallback for comedy skit...")
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
                        {"role": "system", "content": "You write viral relatable workplace comedy dialogues. Output ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1200,
                },
                timeout=60,
            )
            if res.status_code == 200:
                raw_response = res.json()["result"]["response"]
                if isinstance(raw_response, dict):
                    return raw_response
                json_match = re.search(r"\{.*\}", str(raw_response), re.DOTALL)
                return json.loads(json_match.group(0) if json_match else str(raw_response))
        except Exception as e:
            print(f"Cloudflare skit attempt {attempt + 1} error: {e}")
        time.sleep(5)

    raise RuntimeError("All skit generation attempts failed.")


async def _try_edge_tts(text, voice, audio_path):
    communicate = edge_tts.Communicate(text, voice, rate="+8%")
    await communicate.save(audio_path)


def generate_line_audio(text, voice, audio_path):
    clean_text = re.sub(r"[^\w\s.,!?'-]", "", text).strip() or "Excuse me?"
    for attempt in range(3):
        try:
            asyncio.run(_try_edge_tts(clean_text, voice, audio_path))
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 800:
                return
        except Exception as e:
            print(f"Edge-TTS attempt {attempt + 1} ({voice}) failed: {e}")
            time.sleep(2)

    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.save(audio_path)


def _load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_stickman_character(
    draw,
    cx,
    cy,
    scale=1.0,
    facing=1,
    emotion="neutral",
    mouth_open=0.0,
    is_boss=True,
    gesture_phase=0.0,
):
    """Draws a crisp 2D Mercy-J-style expressive stickman with tie/bun, eyebrows, and animated mouth."""
    stroke = max(6, int(10 * scale))
    head_r = int(88 * scale)
    head_cx = cx
    head_cy = cy - int(250 * scale)

    # Torso
    neck_y = head_cy + head_r
    hip_y = neck_y + int(235 * scale)
    draw.line([(cx, neck_y), (cx, hip_y)], fill="#111111", width=stroke)

    # Tie (Speaker A) or Hair Bun / Collar (Speaker B)
    if is_boss:
        tie_pts = [
            (cx, neck_y + int(12 * scale)),
            (cx - int(18 * scale), neck_y + int(125 * scale)),
            (cx, neck_y + int(155 * scale)),
            (cx + int(18 * scale), neck_y + int(125 * scale)),
        ]
        draw.polygon(tie_pts, fill="#D9383A", outline="#111111")
    else:
        # Clean side hair bun & collar V
        bun_x = head_cx - facing * int(head_r * 0.85)
        bun_y = head_cy - int(head_r * 0.45)
        bun_r = int(34 * scale)
        draw.ellipse(
            [bun_x - bun_r, bun_y - bun_r, bun_x + bun_r, bun_y + bun_r],
            fill="#111111",
        )
        draw.line(
            [(cx - int(24 * scale), neck_y + int(26 * scale)), (cx, neck_y + int(52 * scale)), (cx + int(24 * scale), neck_y + int(26 * scale))],
            fill="#2B6CB0",
            width=max(4, int(7 * scale)),
        )

    # Arms with subtle talking gesture
    shoulder_y = neck_y + int(45 * scale)
    wave = int(math.sin(gesture_phase * 2 * math.pi) * 28 * scale) if mouth_open > 0.1 else 0

    if emotion in ("smug", "annoyed") and mouth_open > 0.1:
        # Expressive pointing / gesturing arm toward the other character
        elbow_x = cx + facing * int(85 * scale)
        elbow_y = shoulder_y + int(35 * scale) - wave
        hand_x = cx + facing * int(165 * scale)
        hand_y = shoulder_y - int(25 * scale) - wave
        draw.line([(cx, shoulder_y), (elbow_x, elbow_y), (hand_x, hand_y)], fill="#111111", width=stroke)
        # Other hand on hip
        draw.line(
            [(cx, shoulder_y), (cx - facing * int(75 * scale), shoulder_y + int(65 * scale)), (cx - facing * int(25 * scale), hip_y - int(20 * scale))],
            fill="#111111",
            width=stroke,
        )
    elif emotion == "deadpan":
        # Arms crossed / calm folded posture
        draw.line(
            [(cx - int(65 * scale), shoulder_y + int(65 * scale)), (cx + int(65 * scale), shoulder_y + int(65 * scale))],
            fill="#111111",
            width=stroke,
        )
        draw.line([(cx, shoulder_y), (cx - int(65 * scale), shoulder_y + int(65 * scale))], fill="#111111", width=stroke)
        draw.line([(cx, shoulder_y), (cx + int(65 * scale), shoulder_y + int(65 * scale))], fill="#111111", width=stroke)
    else:
        # Relaxed natural arms
        draw.line(
            [(cx, shoulder_y), (cx + facing * int(60 * scale), shoulder_y + int(95 * scale) - wave)],
            fill="#111111",
            width=stroke,
        )
        draw.line(
            [(cx, shoulder_y), (cx - facing * int(55 * scale), shoulder_y + int(115 * scale))],
            fill="#111111",
            width=stroke,
        )

    # Legs (standing behind desk/counter)
    foot_y = hip_y + int(175 * scale)
    draw.line([(cx, hip_y), (cx - int(48 * scale), foot_y)], fill="#111111", width=stroke)
    draw.line([(cx, hip_y), (cx + int(48 * scale), foot_y)], fill="#111111", width=stroke)

    # Head circle (white fill with bold black outline)
    draw.ellipse(
        [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
        fill="#FFFFFF",
        outline="#111111",
        width=stroke,
    )

    # Eyes (shifted slightly toward facing direction)
    eye_shift = facing * int(18 * scale)
    left_eye_x = head_cx + eye_shift - int(26 * scale)
    right_eye_x = head_cx + eye_shift + int(26 * scale)
    eye_y = head_cy - int(14 * scale)
    eye_r = max(5, int(9 * scale))

    if emotion == "deadpan":
        # Half-lidded unimpressed eyes
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.rectangle([left_eye_x - eye_r - 3, eye_y - eye_r - 3, right_eye_x + eye_r + 3, eye_y], fill="#FFFFFF")
        draw.line([(left_eye_x - 14, eye_y), (left_eye_x + 14, eye_y)], fill="#111111", width=max(4, int(6 * scale)))
        draw.line([(right_eye_x - 14, eye_y), (right_eye_x + 14, eye_y)], fill="#111111", width=max(4, int(6 * scale)))
    else:
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill="#111111")

    # Eyebrows
    brow_w = int(22 * scale)
    brow_y = eye_y - int(22 * scale)
    brow_thick = max(4, int(6 * scale))
    if emotion in ("annoyed", "smug"):
        draw.line([(left_eye_x - brow_w, brow_y - 6), (left_eye_x + brow_w, brow_y + 8)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y + 8), (right_eye_x + brow_w, brow_y - 6)], fill="#111111", width=brow_thick)
    elif emotion == "shocked":
        draw.line([(left_eye_x - brow_w, brow_y - 14), (left_eye_x + brow_w, brow_y - 18)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y - 18), (right_eye_x + brow_w, brow_y - 14)], fill="#111111", width=brow_thick)
    else:
        draw.line([(left_eye_x - brow_w, brow_y), (left_eye_x + brow_w, brow_y)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y), (right_eye_x + brow_w, brow_y)], fill="#111111", width=brow_thick)

    # Mouth (lip-sync open/close + expression)
    mouth_cx = head_cx + eye_shift
    mouth_cy = head_cy + int(34 * scale)
    if mouth_open > 0.15:
        mw = int(24 * scale)
        mh = max(8, int(24 * scale * mouth_open))
        draw.ellipse(
            [mouth_cx - mw, mouth_cy - mh, mouth_cx + mw, mouth_cy + mh],
            fill="#8B1E24",
            outline="#111111",
            width=max(3, int(5 * scale)),
        )
    else:
        if emotion in ("smug", "confident"):
            draw.arc([mouth_cx - 24, mouth_cy - 16, mouth_cx + 24, mouth_cy + 14], start=15, end=165, fill="#111111", width=max(4, int(6 * scale)))
        elif emotion == "annoyed":
            draw.arc([mouth_cx - 22, mouth_cy - 4, mouth_cx + 22, mouth_cy + 22], start=195, end=345, fill="#111111", width=max(4, int(6 * scale)))
        else:
            draw.line([(mouth_cx - 20, mouth_cy), (mouth_cx + 20, mouth_cy)], fill="#111111", width=max(4, int(6 * scale)))


def render_stickman_scene_frame(
    scenario,
    active_speaker,
    emotion,
    mouth_open,
    gesture_phase,
    turn_idx,
    total_turns,
):
    """Renders a crisp 1080x1920 2D office/workplace scene with active speaker camera focus."""
    W, H = 1080, 1920
    img = Image.new("RGB", (W, H), "#F4F6F9")
    draw = ImageDraw.Draw(img)

    # Clean two-tone studio wall + floor
    draw.rectangle([0, 0, W, 1180], fill="#EEF2F7")
    draw.rectangle([0, 1180, W, H], fill="#DCE3EC")
    draw.line([(0, 1180), (W, 1180)], fill="#9AA5B4", width=6)

    # Office window & wall clock prop in background
    draw.rounded_rectangle([120, 210, 460, 560], radius=18, fill="#D8ECFF", outline="#64748B", width=8)
    draw.line([(290, 210), (290, 560)], fill="#64748B", width=8)
    draw.line([(120, 385), (460, 385)], fill="#64748B", width=8)

    # Wall badge / clock prop on top right
    draw.rounded_rectangle([650, 240, 960, 360], radius=20, fill="#FFFFFF", outline="#1E293B", width=7)
    badge_font = _load_font(42)
    draw.text((805, 300), scenario["prop_badge"], fill="#DC2626", font=badge_font, anchor="mm")

    # Role nameplate pills
    pill_font = _load_font(34)
    # Left character (Speaker A)
    a_bg = "#FEE2E2" if active_speaker == "A" else "#E2E8F0"
    a_border = "#DC2626" if active_speaker == "A" else "#64748B"
    draw.rounded_rectangle([120, 620, 420, 690], radius=30, fill=a_bg, outline=a_border, width=5)
    draw.text((270, 655), scenario["role_a"].upper(), fill="#111827", font=pill_font, anchor="mm")

    # Right character (Speaker B)
    b_bg = "#DBEAFE" if active_speaker == "B" else "#E2E8F0"
    b_border = "#2563EB" if active_speaker == "B" else "#64748B"
    draw.rounded_rectangle([660, 620, 960, 690], radius=30, fill=b_bg, outline=b_border, width=5)
    draw.text((810, 655), scenario["role_b"].upper(), fill="#111827", font=pill_font, anchor="mm")

    # Active speaker gets slight dynamic zoom-in while listener stays calm
    scale_a = 1.08 if active_speaker == "A" else 0.96
    scale_b = 1.08 if active_speaker == "B" else 0.96
    emo_a = emotion if active_speaker == "A" else "neutral"
    emo_b = emotion if active_speaker == "B" else "deadpan"
    mouth_a = mouth_open if active_speaker == "A" else 0.0
    mouth_b = mouth_open if active_speaker == "B" else 0.0

    draw_stickman_character(
        draw, cx=285, cy=1140, scale=scale_a, facing=1,
        emotion=emo_a, mouth_open=mouth_a, is_boss=True, gesture_phase=gesture_phase
    )
    draw_stickman_character(
        draw, cx=795, cy=1140, scale=scale_b, facing=-1,
        emotion=emo_b, mouth_open=mouth_b, is_boss=False, gesture_phase=gesture_phase
    )

    # Foreground Office Desk + Laptop Prop
    draw.rounded_rectangle([80, 1080, 1000, 1210], radius=16, fill="#8B5E3C", outline="#3E2723", width=7)
    # Laptop silhouette on right side of desk
    draw.polygon([(640, 1080), (770, 1080), (795, 995), (665, 995)], fill="#CBD5E1", outline="#1E293B")
    draw.rectangle([620, 1072, 780, 1084], fill="#475569")
    # Coffee mug on left side of desk
    draw.rectangle([380, 1030, 425, 1080], fill="#FFFFFF", outline="#1E293B", width=5)

    # Dark caption stage area at bottom
    draw.rounded_rectangle([55, 1245, 1025, 1690], radius=32, fill="#111827", outline="#334155", width=5)

    return np.array(img)


def split_into_subtitle_chunks(text, total_duration):
    words = text.strip().split()
    if not words:
        return []
    chunk_size = 5
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    total_chars = sum(len(c) for c in chunks)

    timed_chunks = []
    current_time = 0.0
    for c in chunks:
        share = (len(c) / total_chars) * total_duration if total_chars > 0 else (total_duration / len(chunks))
        dur = max(0.45, share)
        if current_time + dur > total_duration:
            dur = max(0.15, total_duration - current_time)
        timed_chunks.append((c, current_time, dur))
        current_time += dur
    return timed_chunks


def build_stickman_video(scenario, skit, output_path="stickman_pov.mp4"):
    voice_a, voice_b = random.choice(VOICE_PAIRS)
    dialogue = skit["dialogue"]
    turn_clips = []

    for idx, turn in enumerate(dialogue):
        speaker = turn.get("speaker", "A").upper()
        emotion = turn.get("emotion", "neutral").lower()
        text = turn.get("text", "").strip()
        if not text:
            continue

        aud_file = f"turn_{idx}.mp3"
        voice = voice_a if speaker == "A" else voice_b
        generate_line_audio(text, voice, aud_file)

        audio = AudioFileClip(aud_file)
        dur = audio.duration

        # Create lip-sync animation sequence (alternates closed / half-open / open mouth frames every 0.14s)
        step = 0.14
        mouth_states = [0.0, 0.75, 1.0, 0.45, 0.9, 0.15]
        anim_frames = []
        t_cursor = 0.0
        step_idx = 0
        while t_cursor < dur:
            seg_dur = min(step, dur - t_cursor)
            if seg_dur <= 0.01:
                break
            m_open = mouth_states[step_idx % len(mouth_states)]
            frame_arr = render_stickman_scene_frame(
                scenario=scenario,
                active_speaker=speaker,
                emotion=emotion,
                mouth_open=m_open,
                gesture_phase=(t_cursor / max(dur, 0.1)),
                turn_idx=idx,
                total_turns=len(dialogue),
            )
            anim_frames.append(ImageClip(frame_arr).with_duration(seg_dur))
            t_cursor += seg_dur
            step_idx += 1

        bg_clip = concatenate_videoclips(anim_frames, method="compose")
        layers = [bg_clip]

        speaker_label = scenario["role_a"].upper() if speaker == "A" else scenario["role_b"].upper()
        label_color = "#FCA5A5" if speaker == "A" else "#93C5FD"
        header_txt = (
            TextClip(
                text=f"{speaker_label}:",
                font_size=40,
                color=label_color,
                method="caption",
                size=(880, None),
                text_align="center",
            )
            .with_position(("center", 1285))
            .with_duration(dur)
        )
        layers.append(header_txt)

        for phrase, start_t, sub_dur in split_into_subtitle_chunks(text, dur):
            fade_dur = min(0.10, sub_dur * 0.2)
            txt = (
                TextClip(
                    text=f"\n{phrase}\n",
                    font_size=58,
                    color="#FFFFFF",
                    stroke_color="#000000",
                    stroke_width=2,
                    method="caption",
                    size=(880, None),
                    text_align="center",
                )
                .with_position(("center", 1365))
                .with_start(start_t)
                .with_duration(sub_dur)
                .with_effects([CrossFadeIn(fade_dur), CrossFadeOut(fade_dur)])
            )
            layers.append(txt)

        turn_comp = CompositeVideoClip(layers, size=(1080, 1920)).with_audio(audio)
        turn_clips.append(turn_comp)

    final_video = concatenate_videoclips(turn_clips, method="compose")
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path


def publish_to_all_platforms(public_video_url, caption):
    url = "https://api.buffer.com"
    headers = {
        "Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    org_res = requests.post(
        url,
        headers=headers,
        json={"query": "query { account { organizations { id } } }"},
    ).json()
    if "errors" in org_res:
        raise RuntimeError(f"Buffer auth/org error: {org_res['errors']}")

    org_id = org_res["data"]["account"]["organizations"][0]["id"]
    ch_query = """
    query GetChannels($orgId: OrganizationId!) {
      channels(input: { organizationId: $orgId }) { id name service isDisconnected }
    }
    """
    ch_res = requests.post(
        url,
        headers=headers,
        json={"query": ch_query, "variables": {"orgId": org_id}},
    ).json()
    active_channels = [
        c for c in ch_res.get("data", {}).get("channels", []) if not c.get("isDisconnected")
    ]
    if not active_channels:
        raise RuntimeError(f"No active channels connected in Buffer. Response: {ch_res}")

    create_post_mutation = """
    mutation CreatePost($input: CreatePostInput!) {
      createPost(input: $input) {
        ... on PostActionSuccess { post { id status } }
        ... on MutationError { message }
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
            "assets": [{"video": {"url": public_video_url}}],
        }

        if service == "instagram":
            post_input["metadata"] = {
                "instagram": {"type": "reel", "shouldShareToFeed": True}
            }
        elif service == "youtube":
            yt_title = caption.split("#")[0].strip()[:90] or "POV: Managers Be Like"
            if "#shorts" not in yt_title.lower():
                yt_title = f"{yt_title} #shorts"[:100]
            post_input["metadata"] = {
                "youtube": {"title": yt_title, "categoryId": "23"}
            }

        post_res = requests.post(
            url,
            headers=headers,
            json={"query": create_post_mutation, "variables": {"input": post_input}},
        ).json()
        print(f"Published to {ch['service']} ({ch['name']}):", post_res)


if __name__ == "__main__":
    scenario = pick_scenario()
    print(f"Selected Stickman Comedy Scenario: {scenario['topic']}")
    skit = generate_comedy_skit(scenario)
    video_file = build_stickman_video(scenario, skit)

    upload = cloudinary.uploader.upload(video_file, resource_type="video")
    try:
        publish_to_all_platforms(upload["secure_url"], skit["caption"])
        save_used_topic(scenario["topic"])
        time.sleep(90)
    finally:
        cloudinary.uploader.destroy(upload["public_id"], resource_type="video")
