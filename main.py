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
from faster_whisper import WhisperModel
from google import genai
from moviepy import (
    ImageClip,
    AudioFileClip,
    AudioArrayClip,
    CompositeAudioClip,
    concatenate_videoclips,
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLOUDINARY_URL = os.environ["CLOUDINARY_URL"]
BUFFER_ACCESS_TOKEN = os.environ["BUFFER_ACCESS_TOKEN"]
HISTORY_FILE = "used_topics.json"
SERIES_FILE = "series_state.json"

cloudinary.config(cloudinary_url=CLOUDINARY_URL)

MALE_VOICES = [
    "en-US-AndrewNeural",
    "en-US-BrianNeural",
    "en-US-GuyNeural",
    "en-US-ChristopherNeural",
]

FEMALE_VOICES = [
    "en-US-AvaNeural",
    "en-US-EmmaNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
]

# Emotion-driven vocal delivery settings (rate, pitch, volume)
EMOTION_PROSODY = {
    "angry":     {"rate": "+16%", "pitch": "+5Hz",  "volume": "+15%"},
    "annoyed":   {"rate": "+12%", "pitch": "+2Hz",  "volume": "+5%"},
    "shocked":   {"rate": "+18%", "pitch": "+9Hz",  "volume": "+12%"},
    "hurt":      {"rate": "-4%",  "pitch": "-3Hz",  "volume": "-8%"},
    "nervous":   {"rate": "+15%", "pitch": "+4Hz",  "volume": "-5%"},
    "smug":      {"rate": "+6%",  "pitch": "+2Hz",  "volume": "+4%"},
    "confident": {"rate": "+8%",  "pitch": "-1Hz",  "volume": "+8%"},
    "deadpan":   {"rate": "+4%",  "pitch": "-2Hz",  "volume": "+0%"},
    "neutral":   {"rate": "+9%",  "pitch": "+0Hz",  "volume": "+0%"},
}

CONTENT_PILLARS = [
    {
        "pillar": "Men vs. Women Psychology & Relationship Humor",
        "setting": "living_room",
        "examples": "Why women need emotional validation while men try to fix everything; what 'I'm fine' really feels like; feeling unheard after a long day",
        "allow_multipart": False,
    },
    {
        "pillar": "Workplace Disrespect & Quiet Standing-Up-For-Yourself Saga",
        "setting": "office",
        "examples": "Manager cancels employee's long-planned family vacation PTO last minute; boss humiliates loyal worker in a meeting; unpaid overtime guilt-trip",
        "allow_multipart": True,
    },
    {
        "pillar": "Dating, Loyalty & Modern Relationship Heartbreak/Karma",
        "setting": "restaurant",
        "examples": "Partner hides you from social media while flirting online; first date mocks your honest job until seeing who owns the place; splitting bills unfairly",
        "allow_multipart": True,
    },
    {
        "pillar": "Family, Roommate & Neighbor Boundaries",
        "setting": "living_room",
        "examples": "Parents always favor the golden-child sibling until they need help; roommate takes advantage of your kindness; entitled neighbor blocks driveway",
        "allow_multipart": True,
    },
    {
        "pillar": "Entitled Customer vs. Exhausted Service Worker",
        "setting": "counter",
        "examples": "Customer screams at teenage cashier over expired coupon; regular customer defends humiliated worker; fake 'I'll get you fired' threat",
        "allow_multipart": False,
    },
    {
        "pillar": "Landlord, HOA & Unfair Contract Showdowns",
        "setting": "office",
        "examples": "Landlord tries to steal deposit from a single parent; HOA fines family for kids' sidewalk chalk; gym refuses to cancel after injury",
        "allow_multipart": True,
    },
    {
        "pillar": "School, Teacher & Underestimated Student Moments",
        "setting": "classroom",
        "examples": "Teacher doubts hardworking student in front of class; group project teammates dump all work on one quiet student",
        "allow_multipart": False,
    },
]

WALL_THEMES = [
    {"wall": "#EEF2F7", "floor": "#DCE3EC", "accent_a": "#DC2626", "accent_b": "#2563EB"},
    {"wall": "#FDF6EC", "floor": "#EDE0CE", "accent_a": "#D97706", "accent_b": "#0F766E"},
    {"wall": "#F3F0FA", "floor": "#E2DCF2", "accent_a": "#7C3AED", "accent_b": "#059669"},
    {"wall": "#F0F7F4", "floor": "#DCEAE4", "accent_a": "#BE123C", "accent_b": "#1D4ED8"},
]

_WHISPER_MODEL = None


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


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
    if topic and topic not in used:
        used.append(topic)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"used_topics": used}, f, indent=2)


def load_series_state():
    if os.path.exists(SERIES_FILE):
        try:
            with open(SERIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("active"):
                    return data
        except Exception:
            pass
    return None


def save_series_state(state):
    with open(SERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _call_llm_json(prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash"]

    for attempt in range(4):
        model_name = models_to_try[attempt % len(models_to_try)]
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            clean_json = re.sub(r"```json|```", "", response.text).strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Gemini attempt {attempt + 1} ({model_name}) failed: {e}")
            time.sleep(8)

    print("Switching to Cloudflare Llama 3.3 70B fallback...")
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
                        {"role": "system", "content": "Output ONLY valid raw JSON with no markdown or extra commentary."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1500,
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
            print(f"Cloudflare attempt {attempt + 1} error: {e}")
        time.sleep(5)

    raise RuntimeError("All LLM script generation attempts failed.")


def _infer_genders(role_a, role_b):
    ra = role_a.lower()
    rb = role_b.lower()
    female_hints = ("her", "wife", "girlfriend", "mom", "mother", "sister", "woman", "lady", "daughter")
    male_hints = ("him", "husband", "boyfriend", "dad", "father", "brother", "guy", "man", "bro", "son")

    if any(h in ra for h in female_hints):
        gen_a = "F"
    elif any(h in ra for h in male_hints):
        gen_a = "M"
    else:
        gen_a = random.choice(["M", "F"])

    if any(h in rb for h in female_hints):
        gen_b = "F"
    elif any(h in rb for h in male_hints):
        gen_b = "M"
    else:
        gen_b = "F" if gen_a == "M" else "M"

    return gen_a, gen_b


def _pick_gendered_voices(gen_a, gen_b):
    va = random.choice(FEMALE_VOICES if gen_a == "F" else MALE_VOICES)
    pool_b = FEMALE_VOICES if gen_b == "F" else MALE_VOICES
    choices_b = [v for v in pool_b if v != va] or pool_b
    vb = random.choice(choices_b)
    return [va, vb]


def generate_skit_episode():
    series = load_series_state()
    used_topics = load_used_topics()

    if series:
        part_num = series["current_part"] + 1
        total_parts = series["total_parts"]
        is_finale = (part_num >= total_parts)
        past_episodes = "\n\n".join(
            f"PART {i+1} SUMMARY:\n{ep}" for i, ep in enumerate(series["history_scripts"])
        )

        ending_instruction = (
            f"This is PART {part_num} of {total_parts} (THE EMOTIONAL FINALE). Deliver the deeply satisfying truth, vindication, and a memorable closing line viewers will cheer for!"
            if is_finale
            else f"This is PART {part_num} of {total_parts} (MIDDLE CHAPTER). Raise the emotional stakes, reveal a painful or shocking truth, and end on a cliffhanger teasing Part {part_num + 1}!"
        )

        prompt = f"""
        Continue this emotional, relatable stickman story in the style of "Mercy J Animation":
        SERIES TOPIC: "{series['topic']}"
        CATEGORY: "{series['pillar']}"
        SPEAKER A: "{series['role_a']}" | SPEAKER B: "{series['role_b']}"

        WHAT HAPPENED SO FAR:
        {past_episodes}

        PART {series['current_part']} ENDED ON THIS LINE:
        "{series['last_line']}"

        WRITING RULES FOR MAXIMUM AUDIENCE CONNECTION:
        1. Pick up immediately from where Part {series['current_part']} left off—no recap intro.
        2. {ending_instruction}
        3. Write 8 to 10 alternating dialogue turns (12 to 24 words per turn) using natural, heartfelt, human American English—avoid stiff corporate buzzwords.
        4. Include genuine emotional shifts: frustration, vulnerability, disbelief, quiet dignity, and wit.
        5. Assign each line the exact matching emotion from:
           ["angry", "annoyed", "shocked", "hurt", "nervous", "smug", "confident", "deadpan", "neutral"].

        Return ONLY valid JSON:
        {{
          "prop_badge": "PART {part_num}/{total_parts}",
          "caption": "POV: {series['caption_hook']} (Part {part_num}/{total_parts}) 😭🔥 #stickmancomedy #relatable #pov #part{part_num} #shorts",
          "dialogue": [
            {{"speaker": "A", "emotion": "angry", "text": "..."}},
            {{"speaker": "B", "emotion": "hurt", "text": "..."}}
          ]
        }}
        """
        ep = _call_llm_json(prompt)
        ep["topic"] = series["topic"]
        ep["pillar"] = series["pillar"]
        ep["setting"] = series["setting"]
        ep["role_a"] = series["role_a"]
        ep["role_b"] = series["role_b"]
        ep["gen_a"] = series.get("gen_a", "M")
        ep["gen_b"] = series.get("gen_b", "F")
        ep["voice_pair"] = series["voice_pair"]
        ep["theme"] = series["theme"]
        ep["part_num"] = part_num
        ep["total_parts"] = total_parts
        ep["caption_hook"] = series["caption_hook"]
        ep["prop_badge"] = f"PART {part_num} OF {total_parts}"
        return ep

    pillar_obj = random.choice(CONTENT_PILLARS)
    make_multipart = pillar_obj["allow_multipart"] and (random.random() < 0.55)
    total_parts = random.choice([3, 4, 5]) if make_multipart else 1
    recent_blocklist = "\n".join(f"- {t}" for t in used_topics[-150:]) or "- None yet"

    part_instruction = (
        f"This is PART 1 of a {total_parts}-PART emotional drama/comedy series. Make the audience immediately feel for Speaker B, escalate the unfairness, and end on a cliffhanger!"
        if total_parts > 1
        else "This is a COMPLETE STANDALONE 1-part emotional comedy/psychology Short. Start with a relatable everyday sting, build real emotion, and land a deeply satisfying punchline."
    )

    prompt = f"""
    Create a BRAND-NEW, deeply relatable 55-second two-person stickman story modeled after "Mercy J Animation" Shorts.
    CHOSEN CATEGORY: "{pillar_obj['pillar']}"
    INSPIRATION EXAMPLES (invent a fresh original angle): {pillar_obj['examples']}

    NEVER REPEAT ANY OF THESE PREVIOUSLY USED TOPICS:
    {recent_blocklist}

    EMOTIONAL & HUMAN CONNECTION RULES:
    1. {part_instruction}
    2. Write like real humans talking—raw, relatable feelings (feeling taken for granted, standing up for yourself, relationship honesty, quiet self-respect) mixed with sharp humor.
    3. Avoid robotic corporate jargon. Use natural contractions ("I'm", "you've", "honestly", "look").
    4. Write 8 to 10 alternating dialogue turns between Speaker A and Speaker B (12 to 24 words per turn).
    5. Create a clear emotional arc across the 8-10 lines (e.g., smug -> hurt -> shocked -> confident mic-drop).
    6. Assign each turn the exact emotion from:
       ["angry", "annoyed", "shocked", "hurt", "nervous", "smug", "confident", "deadpan", "neutral"].

    Return ONLY valid JSON:
    {{
      "topic": "Unique 8-to-12-word summary of this exact story",
      "caption_hook": "Emotional relatable 5-to-9-word hook without hashtags",
      "role_a": "Role name (max 13 chars, e.g. Her, Manager, Dad, Customer)",
      "role_b": "Role name (max 13 chars, e.g. Him, Employee, Son, Cashier)",
      "prop_badge": "SHORT WALL SIGN (max 12 chars, e.g. PART 1 OF {total_parts}, UNFAIR, 11:58 PM, TRUTH)",
      "caption": "POV: Catchy Emotional Hook 😭🔥 #stickmancomedy #relatable #pov #storytime #shorts",
      "dialogue": [
        {{"speaker": "A", "emotion": "smug", "text": "..."}},
        {{"speaker": "B", "emotion": "hurt", "text": "..."}}
      ]
    }}
    """

    for _ in range(3):
        ep = _call_llm_json(prompt)
        topic = ep.get("topic", "").strip()
        if (
            topic
            and topic.lower() not in {u.lower() for u in used_topics}
            and ep.get("dialogue")
            and len(ep["dialogue"]) >= 4
        ):
            ep["pillar"] = pillar_obj["pillar"]
            ep["setting"] = pillar_obj["setting"]
            ep["role_a"] = str(ep.get("role_a", "Speaker A"))[:14]
            ep["role_b"] = str(ep.get("role_b", "Speaker B"))[:14]
            gen_a, gen_b = _infer_genders(ep["role_a"], ep["role_b"])
            ep["gen_a"] = gen_a
            ep["gen_b"] = gen_b
            ep["voice_pair"] = _pick_gendered_voices(gen_a, gen_b)
            ep["prop_badge"] = (
                f"PART 1 OF {total_parts}"
                if total_parts > 1
                else str(ep.get("prop_badge", "POV"))[:13].upper()
            )
            ep["theme"] = random.choice(WALL_THEMES)
            ep["part_num"] = 1
            ep["total_parts"] = total_parts
            ep["caption_hook"] = ep.get("caption_hook", topic)[:65]
            if total_parts > 1 and "part 1" not in ep["caption"].lower():
                ep["caption"] = f"POV: {ep['caption_hook']} (Part 1/{total_parts}) 😭🔥 #stickmancomedy #relatable #pov #part1 #shorts"
            return ep

    raise RuntimeError("Failed to generate a unique non-repeating episode.")


async def _try_edge_tts(text, voice, emotion, audio_path):
    prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=prosody["rate"],
        pitch=prosody["pitch"],
        volume=prosody["volume"],
    )
    await communicate.save(audio_path)


def generate_line_audio(text, voice, emotion, audio_path):
    clean_text = re.sub(r"[^\w\s.,!?'-]", "", text).strip() or "Wait a second."
    for attempt in range(3):
        try:
            asyncio.run(_try_edge_tts(clean_text, voice, emotion, audio_path))
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 800:
                return
        except Exception as e:
            print(f"Edge-TTS attempt {attempt + 1} ({voice}, {emotion}) failed: {e}")
            time.sleep(2)

    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.save(audio_path)


def create_emotional_bg_music(duration, fps=44100):
    """Synthesizes a warm, emotional lo-fi piano/pad chord progression at gentle background volume."""
    t = np.linspace(0, duration, int(fps * duration), endpoint=False)
    # Gentle Am - F - C - G emotional chord progression (4s per chord)
    chords = [
        (220.00, 261.63, 329.63),  # Am
        (174.61, 220.00, 261.63),  # F
        (261.63, 329.63, 392.00),  # C
        (196.00, 246.94, 293.66),  # G
    ]
    signal = np.zeros_like(t)
    chord_len = 3.6
    for i, (f1, f2, f3) in enumerate(chords):
        mask = ((t // chord_len).astype(int) % len(chords)) == i
        local_t = t % chord_len
        env = np.sin(np.pi * np.clip(local_t / chord_len, 0, 1)) ** 0.7
        pad = (
            0.45 * np.sin(2 * np.pi * f1 * t)
            + 0.35 * np.sin(2 * np.pi * f2 * t)
            + 0.25 * np.sin(2 * np.pi * f3 * t)
            + 0.12 * np.sin(2 * np.pi * (f1 * 0.5) * t)
        )
        signal += np.where(mask, pad * env, 0.0)

    mix = 0.032 * signal
    stereo = np.column_stack((mix, mix)).astype(np.float32)
    return AudioArrayClip(stereo, fps=fps)


def get_word_timings(audio_path, fallback_text, total_duration):
    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=False)
        timed_words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    tok = w.word.strip()
                    if tok:
                        timed_words.append((tok, float(w.start), float(w.end)))
        if len(timed_words) >= 2:
            return timed_words
    except Exception as e:
        print(f"Whisper word timing fallback ({e})")

    words = fallback_text.strip().split()
    if not words:
        return []
    total_chars = sum(len(w) for w in words) or 1
    cur = 0.0
    out = []
    for w in words:
        dur = max(0.12, (len(w) / total_chars) * total_duration)
        out.append((w, cur, min(total_duration, cur + dur)))
        cur += dur
    return out


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
    gender="M",
    accent_color="#DC2626",
    gesture_phase=0.0,
    is_speaking=False,
):
    stroke = max(6, int(10 * scale))
    # Dynamic emotional body language: lean in when angry/confident, slump slightly when hurt
    lean_x = 0
    slump_y = 0
    shake_x = 0
    if is_speaking and emotion in ("angry", "annoyed"):
        lean_x = facing * int(18 * scale)
        shake_x = int(math.sin(gesture_phase * 18 * math.pi) * 5 * scale)
    elif emotion == "hurt":
        slump_y = int(16 * scale)
    elif emotion == "shocked":
        shake_x = int(math.sin(gesture_phase * 14 * math.pi) * 4 * scale)

    head_r = int(88 * scale)
    head_cx = cx + lean_x + shake_x
    head_cy = cy - int(250 * scale) + slump_y

    neck_y = head_cy + head_r
    hip_y = cy - int(15 * scale)
    draw.line([(head_cx, neck_y), (cx, hip_y)], fill="#111111", width=stroke)

    if gender == "M":
        tie_pts = [
            (head_cx, neck_y + int(12 * scale)),
            (head_cx - int(18 * scale), neck_y + int(125 * scale)),
            (head_cx, neck_y + int(155 * scale)),
            (head_cx + int(18 * scale), neck_y + int(125 * scale)),
        ]
        draw.polygon(tie_pts, fill=accent_color, outline="#111111")
    else:
        bun_x = head_cx - facing * int(head_r * 0.85)
        bun_y = head_cy - int(head_r * 0.45)
        bun_r = int(34 * scale)
        draw.ellipse(
            [bun_x - bun_r, bun_y - bun_r, bun_x + bun_r, bun_y + bun_r],
            fill="#111111",
        )
        draw.line(
            [
                (head_cx - int(24 * scale), neck_y + int(26 * scale)),
                (head_cx, neck_y + int(52 * scale)),
                (head_cx + int(24 * scale), neck_y + int(26 * scale)),
            ],
            fill=accent_color,
            width=max(4, int(7 * scale)),
        )

    shoulder_x = head_cx
    shoulder_y = neck_y + int(45 * scale)
    wave = int(math.sin(gesture_phase * 2 * math.pi) * 28 * scale) if mouth_open > 0.1 else 0

    if emotion in ("angry", "smug", "annoyed") and mouth_open > 0.1:
        elbow_x = shoulder_x + facing * int(85 * scale)
        elbow_y = shoulder_y + int(30 * scale) - wave
        hand_x = shoulder_x + facing * int(170 * scale)
        hand_y = shoulder_y - int(30 * scale) - wave
        draw.line([(shoulder_x, shoulder_y), (elbow_x, elbow_y), (hand_x, hand_y)], fill="#111111", width=stroke)
        draw.line(
            [
                (shoulder_x, shoulder_y),
                (shoulder_x - facing * int(75 * scale), shoulder_y + int(65 * scale)),
                (cx - facing * int(25 * scale), hip_y - int(20 * scale)),
            ],
            fill="#111111",
            width=stroke,
        )
    elif emotion == "shocked":
        # Both hands raised near head in disbelief
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x - int(75 * scale), shoulder_y - int(15 * scale)), (head_cx - int(95 * scale), head_cy + int(10 * scale))],
            fill="#111111",
            width=stroke,
        )
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x + int(75 * scale), shoulder_y - int(15 * scale)), (head_cx + int(95 * scale), head_cy + int(10 * scale))],
            fill="#111111",
            width=stroke,
        )
    elif emotion == "hurt":
        # Hand placed gently over chest/heart
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x + facing * int(45 * scale), shoulder_y + int(55 * scale)), (shoulder_x + facing * int(10 * scale), shoulder_y + int(35 * scale))],
            fill="#111111",
            width=stroke,
        )
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x - facing * int(45 * scale), shoulder_y + int(115 * scale))],
            fill="#111111",
            width=stroke,
        )
    elif emotion in ("deadpan", "confident"):
        draw.line(
            [(shoulder_x - int(65 * scale), shoulder_y + int(65 * scale)), (shoulder_x + int(65 * scale), shoulder_y + int(65 * scale))],
            fill="#111111",
            width=stroke,
        )
        draw.line([(shoulder_x, shoulder_y), (shoulder_x - int(65 * scale), shoulder_y + int(65 * scale))], fill="#111111", width=stroke)
        draw.line([(shoulder_x, shoulder_y), (shoulder_x + int(65 * scale), shoulder_y + int(65 * scale))], fill="#111111", width=stroke)
    else:
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x + facing * int(60 * scale), shoulder_y + int(95 * scale) - wave)],
            fill="#111111",
            width=stroke,
        )
        draw.line(
            [(shoulder_x, shoulder_y), (shoulder_x - facing * int(55 * scale), shoulder_y + int(115 * scale))],
            fill="#111111",
            width=stroke,
        )

    foot_y = hip_y + int(175 * scale)
    draw.line([(cx, hip_y), (cx - int(48 * scale), foot_y)], fill="#111111", width=stroke)
    draw.line([(cx, hip_y), (cx + int(48 * scale), foot_y)], fill="#111111", width=stroke)

    # Head circle
    draw.ellipse(
        [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
        fill="#FFFFFF",
        outline="#111111",
        width=stroke,
    )

    eye_shift = facing * int(18 * scale)
    left_eye_x = head_cx + eye_shift - int(26 * scale)
    right_eye_x = head_cx + eye_shift + int(26 * scale)
    eye_y = head_cy - int(14 * scale)
    eye_r = max(5, int(9 * scale))

    # Blush / red cheeks for angry or hurt emotions
    if emotion in ("angry", "hurt"):
        cheek_color = "#FCA5A5" if emotion == "angry" else "#FBCFE8"
        draw.ellipse([left_eye_x - 16, eye_y + 14, left_eye_x + 12, eye_y + 30], fill=cheek_color)
        draw.ellipse([right_eye_x - 12, eye_y + 14, right_eye_x + 16, eye_y + 30], fill=cheek_color)

    # Eyes
    if emotion == "deadpan":
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.rectangle([left_eye_x - eye_r - 3, eye_y - eye_r - 3, right_eye_x + eye_r + 3, eye_y], fill="#FFFFFF")
        draw.line([(left_eye_x - 14, eye_y), (left_eye_x + 14, eye_y)], fill="#111111", width=max(4, int(6 * scale)))
        draw.line([(right_eye_x - 14, eye_y), (right_eye_x + 14, eye_y)], fill="#111111", width=max(4, int(6 * scale)))
    elif emotion == "shocked":
        big_r = int(15 * scale)
        draw.ellipse([left_eye_x - big_r, eye_y - big_r, left_eye_x + big_r, eye_y + big_r], fill="#FFFFFF", outline="#111111", width=4)
        draw.ellipse([right_eye_x - big_r, eye_y - big_r, right_eye_x + big_r, eye_y + big_r], fill="#FFFFFF", outline="#111111", width=4)
        draw.ellipse([left_eye_x - 5, eye_y - 5, left_eye_x + 5, eye_y + 5], fill="#111111")
        draw.ellipse([right_eye_x - 5, eye_y - 5, right_eye_x + 5, eye_y + 5], fill="#111111")
    elif emotion == "hurt":
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        # White shine + small blue tear drop
        draw.ellipse([left_eye_x - 4, eye_y - 5, left_eye_x + 1, eye_y], fill="#FFFFFF")
        draw.ellipse([right_eye_x - 4, eye_y - 5, right_eye_x + 1, eye_y], fill="#FFFFFF")
        tear_x = right_eye_x + int(10 * scale)
        tear_y = eye_y + int(18 * scale)
        draw.ellipse([tear_x - 5, tear_y - 7, tear_x + 5, tear_y + 7], fill="#38BDF8")
    else:
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill="#111111")
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill="#111111")

    # Expressive Eyebrows
    brow_w = int(23 * scale)
    brow_y = eye_y - int(22 * scale)
    brow_thick = max(4, int(6 * scale))
    if emotion in ("angry", "annoyed", "smug"):
        draw.line([(left_eye_x - brow_w, brow_y - 8), (left_eye_x + brow_w, brow_y + 10)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y + 10), (right_eye_x + brow_w, brow_y - 8)], fill="#111111", width=brow_thick)
    elif emotion == "hurt":
        # Sad slanted inner-up eyebrows
        draw.line([(left_eye_x - brow_w, brow_y + 8), (left_eye_x + brow_w, brow_y - 8)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y - 8), (right_eye_x + brow_w, brow_y + 8)], fill="#111111", width=brow_thick)
    elif emotion in ("shocked", "nervous"):
        draw.line([(left_eye_x - brow_w, brow_y - 14), (left_eye_x + brow_w, brow_y - 18)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y - 18), (right_eye_x + brow_w, brow_y - 14)], fill="#111111", width=brow_thick)
    else:
        draw.line([(left_eye_x - brow_w, brow_y), (left_eye_x + brow_w, brow_y)], fill="#111111", width=brow_thick)
        draw.line([(right_eye_x - brow_w, brow_y), (right_eye_x + brow_w, brow_y)], fill="#111111", width=brow_thick)

    # Comic anger vein or nervous sweat drop on forehead
    if emotion == "angry":
        vx = head_cx - facing * int(42 * scale)
        vy = head_cy - int(45 * scale)
        draw.arc([vx - 14, vy - 14, vx, vy], start=0, end=90, fill="#DC2626", width=4)
        draw.arc([vx + 4, vy - 14, vx + 18, vy], start=90, end=180, fill="#DC2626", width=4)
        draw.arc([vx - 14, vy + 4, vx, vy + 18], start=270, end=360, fill="#DC2626", width=4)
        draw.arc([vx + 4, vy + 4, vx + 18, vy + 18], start=180, end=270, fill="#DC2626", width=4)
    elif emotion in ("nervous", "shocked"):
        sx = head_cx - facing * int(55 * scale)
        sy = head_cy - int(25 * scale)
        draw.polygon([(sx, sy - 14), (sx - 8, sy + 6), (sx + 8, sy + 6)], fill="#38BDF8")
        draw.ellipse([sx - 8, sy, sx + 8, sy + 14], fill="#38BDF8")

    # Mouth (lip-sync + emotional expression)
    mouth_cx = head_cx + eye_shift
    mouth_cy = head_cy + int(34 * scale)
    if mouth_open > 0.15:
        mw = int(26 * scale) if emotion in ("angry", "shocked") else int(22 * scale)
        mh = max(8, int(26 * scale * mouth_open))
        draw.ellipse(
            [mouth_cx - mw, mouth_cy - mh, mouth_cx + mw, mouth_cy + mh],
            fill="#8B1E24",
            outline="#111111",
            width=max(3, int(5 * scale)),
        )
    else:
        if emotion in ("smug", "confident"):
            draw.arc([mouth_cx - 24, mouth_cy - 16, mouth_cx + 24, mouth_cy + 14], start=15, end=165, fill="#111111", width=max(4, int(6 * scale)))
        elif emotion in ("annoyed", "angry", "hurt"):
            draw.arc([mouth_cx - 22, mouth_cy - 4, mouth_cx + 22, mouth_cy + 22], start=195, end=345, fill="#111111", width=max(4, int(6 * scale)))
        else:
            draw.line([(mouth_cx - 20, mouth_cy), (mouth_cx + 20, mouth_cy)], fill="#111111", width=max(4, int(6 * scale)))


def draw_setting_background(draw, setting, theme):
    W, H = 1080, 1920
    draw.rectangle([0, 0, W, 1180], fill=theme["wall"])
    draw.rectangle([0, 1180, W, H], fill=theme["floor"])
    draw.line([(0, 1180), (W, 1180)], fill="#9AA5B4", width=6)

    if setting == "living_room":
        draw.rounded_rectangle([130, 220, 440, 480], radius=14, fill="#FEF3C7", outline="#78350F", width=8)
        draw.ellipse([235, 290, 335, 390], fill="#F59E0B")
    elif setting == "restaurant":
        draw.line([(280, 0), (280, 230)], fill="#1E293B", width=6)
        draw.pieslice([210, 210, 350, 310], start=180, end=360, fill="#F59E0B", outline="#1E293B", width=6)
    elif setting == "classroom":
        draw.rounded_rectangle([110, 200, 520, 520], radius=14, fill="#14532D", outline="#78350F", width=10)
    else:
        draw.rounded_rectangle([120, 210, 460, 560], radius=18, fill="#D8ECFF", outline="#64748B", width=8)
        draw.line([(290, 210), (290, 560)], fill="#64748B", width=8)
        draw.line([(120, 385), (460, 385)], fill="#64748B", width=8)


def draw_setting_foreground(draw, setting):
    if setting == "living_room":
        draw.rounded_rectangle([110, 990, 970, 1190], radius=28, fill="#475569", outline="#1E293B", width=7)
        draw.rounded_rectangle([260, 1120, 820, 1210], radius=14, fill="#92400E", outline="#451A03", width=6)
    elif setting == "restaurant":
        draw.rounded_rectangle([220, 1080, 860, 1210], radius=18, fill="#7C2D12", outline="#3E2723", width=7)
        draw.rectangle([410, 1015, 445, 1080], fill="#F87171", outline="#1E293B", width=4)
        draw.rectangle([635, 1015, 670, 1080], fill="#38BDF8", outline="#1E293B", width=4)
    elif setting == "counter":
        draw.rounded_rectangle([70, 1050, 1010, 1215], radius=14, fill="#334155", outline="#0F172A", width=7)
        draw.rectangle([650, 975, 790, 1050], fill="#94A3B8", outline="#1E293B", width=5)
    else:
        draw.rounded_rectangle([80, 1080, 1000, 1210], radius=16, fill="#8B5E3C", outline="#3E2723", width=7)
        draw.polygon([(640, 1080), (770, 1080), (795, 995), (665, 995)], fill="#CBD5E1", outline="#1E293B")
        draw.rectangle([620, 1072, 780, 1084], fill="#475569")
        draw.rectangle([380, 1030, 425, 1080], fill="#FFFFFF", outline="#1E293B", width=5)


def draw_dynamic_caption_box(draw, ep, active_speaker, chunk_words, active_word_idx):
    draw.rounded_rectangle([65, 1255, 1015, 1625], radius=34, fill="#0F172A", outline="#334155", width=5)

    speaker_label = ep["role_a"].upper() if active_speaker == "A" else ep["role_b"].upper()
    pill_fill = "#DC2626" if active_speaker == "A" else "#2563EB"
    draw.rounded_rectangle([370, 1230, 710, 1292], radius=24, fill=pill_fill, outline="#FFFFFF", width=4)
    speaker_font = _load_font(32)
    draw.text((540, 1261), speaker_label, fill="#FFFFFF", font=speaker_font, anchor="mm")

    if not chunk_words:
        return

    sub_font = _load_font(62)
    lines = []
    current_line = []
    for idx, w in enumerate(chunk_words):
        current_line.append((idx, w.upper()))
        if len(current_line) == 3:
            lines.append(current_line)
            current_line = []
    if current_line:
        lines.append(current_line)

    start_y = 1390 if len(lines) == 2 else 1445
    line_height = 88

    for line_idx, line_items in enumerate(lines[:2]):
        y_pos = start_y + line_idx * line_height
        spaced_words = [w for _, w in line_items]
        widths = [draw.textbbox((0, 0), w, font=sub_font)[2] for w in spaced_words]
        space_w = 24
        total_w = sum(widths) + space_w * (len(spaced_words) - 1)
        x_cursor = (1080 - total_w) // 2

        for (orig_idx, word_str), w_px in zip(line_items, widths):
            is_active = (orig_idx == active_word_idx)
            if is_active:
                draw.rounded_rectangle(
                    [x_cursor - 12, y_pos - 42, x_cursor + w_px + 12, y_pos + 40],
                    radius=14,
                    fill="#FACC15",
                )
                txt_color = "#0F172A"
            else:
                txt_color = "#FFFFFF"

            draw.text((x_cursor, y_pos), word_str, fill=txt_color, font=sub_font, anchor="lm")
            x_cursor += w_px + space_w


def render_stickman_scene_frame(
    ep,
    active_speaker,
    emotion,
    listener_emotion,
    mouth_open,
    gesture_phase,
    chunk_words=None,
    active_word_idx=-1,
):
    img = Image.new("RGB", (1080, 1920), ep["theme"]["wall"])
    draw = ImageDraw.Draw(img)

    draw_setting_background(draw, ep["setting"], ep["theme"])

    draw.rounded_rectangle([610, 235, 990, 360], radius=20, fill="#FFFFFF", outline="#1E293B", width=7)
    badge_font = _load_font(38)
    draw.text((800, 297), ep["prop_badge"], fill="#DC2626", font=badge_font, anchor="mm")

    pill_font = _load_font(34)
    a_bg = "#FEE2E2" if active_speaker == "A" else "#E2E8F0"
    a_border = "#DC2626" if active_speaker == "A" else "#64748B"
    draw.rounded_rectangle([120, 620, 420, 690], radius=30, fill=a_bg, outline=a_border, width=5)
    draw.text((270, 655), ep["role_a"].upper(), fill="#111827", font=pill_font, anchor="mm")

    b_bg = "#DBEAFE" if active_speaker == "B" else "#E2E8F0"
    b_border = "#2563EB" if active_speaker == "B" else "#64748B"
    draw.rounded_rectangle([660, 620, 960, 690], radius=30, fill=b_bg, outline=b_border, width=5)
    draw.text((810, 655), ep["role_b"].upper(), fill="#111827", font=pill_font, anchor="mm")

    scale_a = 1.09 if active_speaker == "A" else 0.96
    scale_b = 1.09 if active_speaker == "B" else 0.96
    emo_a = emotion if active_speaker == "A" else listener_emotion
    emo_b = emotion if active_speaker == "B" else listener_emotion
    mouth_a = mouth_open if active_speaker == "A" else 0.0
    mouth_b = mouth_open if active_speaker == "B" else 0.0

    draw_stickman_character(
        draw, cx=285, cy=1140, scale=scale_a, facing=1,
        emotion=emo_a, mouth_open=mouth_a, gender=ep.get("gen_a", "M"),
        accent_color=ep["theme"]["accent_a"], gesture_phase=gesture_phase,
        is_speaking=(active_speaker == "A"),
    )
    draw_stickman_character(
        draw, cx=795, cy=1140, scale=scale_b, facing=-1,
        emotion=emo_b, mouth_open=mouth_b, gender=ep.get("gen_b", "F"),
        accent_color=ep["theme"]["accent_b"], gesture_phase=gesture_phase,
        is_speaking=(active_speaker == "B"),
    )

    draw_setting_foreground(draw, ep["setting"])
    draw_dynamic_caption_box(draw, ep, active_speaker, chunk_words or [], active_word_idx)
    return np.array(img)


def _reaction_emotion_for_listener(speaker_emotion):
    """Makes the listening character react emotionally instead of freezing blankly."""
    reactions = {
        "angry": "shocked",
        "smug": "annoyed",
        "hurt": "smug",
        "confident": "shocked",
        "shocked": "confident",
    }
    return reactions.get(speaker_emotion, "deadpan")


def build_stickman_video(ep, output_path="stickman_pov.mp4"):
    voice_a, voice_b = ep["voice_pair"]
    dialogue = ep["dialogue"]
    turn_clips = []

    for idx, turn in enumerate(dialogue):
        speaker = turn.get("speaker", "A").upper()
        emotion = turn.get("emotion", "neutral").lower()
        if emotion not in EMOTION_PROSODY:
            emotion = "neutral"
        listener_emo = _reaction_emotion_for_listener(emotion)
        text = turn.get("text", "").strip()
        if not text:
            continue

        aud_file = f"turn_{idx}.mp3"
        voice = voice_a if speaker == "A" else voice_b
        generate_line_audio(text, voice, emotion, aud_file)

        audio = AudioFileClip(aud_file)
        dur = audio.duration
        word_timings = get_word_timings(aud_file, text, dur)

        burst_size = 4
        bursts = []
        for i in range(0, len(word_timings), burst_size):
            group = word_timings[i : i + burst_size]
            b_start = group[0][1] if i > 0 else 0.0
            b_end = word_timings[i + burst_size][1] if (i + burst_size) < len(word_timings) else dur
            bursts.append((group, b_start, max(b_end, b_start + 0.12)))

        step = 0.12
        mouth_states = [0.0, 0.85, 1.0, 0.4, 0.9, 0.15]
        anim_frames = []
        t_cursor = 0.0
        step_idx = 0

        while t_cursor < dur:
            seg_dur = min(step, dur - t_cursor)
            if seg_dur <= 0.01:
                break

            mid_t = t_cursor + seg_dur * 0.5
            active_words = []
            active_word_idx = -1
            for group, b_start, b_end in bursts:
                if b_start <= mid_t <= b_end:
                    active_words = [w[0] for w in group]
                    for w_i, (_, w_s, w_e) in enumerate(group):
                        if w_s <= mid_t <= max(w_e, w_s + 0.08):
                            active_word_idx = w_i
                            break
                    if active_word_idx == -1 and group:
                        active_word_idx = min(
                            len(group) - 1,
                            max(0, int(((mid_t - b_start) / max(b_end - b_start, 0.05)) * len(group))),
                        )
                    break

            m_open = mouth_states[step_idx % len(mouth_states)] if active_word_idx >= 0 else 0.0
            frame_arr = render_stickman_scene_frame(
                ep=ep,
                active_speaker=speaker,
                emotion=emotion,
                listener_emotion=listener_emo,
                mouth_open=m_open,
                gesture_phase=(t_cursor / max(dur, 0.1)),
                chunk_words=active_words,
                active_word_idx=active_word_idx,
            )
            anim_frames.append(ImageClip(frame_arr).with_duration(seg_dur))
            t_cursor += seg_dur
            step_idx += 1

        turn_comp = concatenate_videoclips(anim_frames, method="compose").with_audio(audio)
        turn_clips.append(turn_comp)

    video = concatenate_videoclips(turn_clips, method="compose")
    bg_music = create_emotional_bg_music(video.duration)
    final_audio = CompositeAudioClip([video.audio, bg_music])
    final_video = video.with_audio(final_audio)

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
            yt_title = caption.split("#")[0].strip()[:90] or "POV: Stickman Comedy"
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
    ep = generate_skit_episode()
    print(f"Episode: {ep['topic']} | Part {ep['part_num']}/{ep['total_parts']} | Voices: {ep['voice_pair']}")
    video_file = build_stickman_video(ep)

    upload = cloudinary.uploader.upload(video_file, resource_type="video")
    try:
        publish_to_all_platforms(upload["secure_url"], ep["caption"])
        save_used_topic(ep["topic"])

        if ep["part_num"] < ep["total_parts"]:
            prev_state = load_series_state()
            history_scripts = prev_state["history_scripts"] if prev_state else []
            history_scripts.append(" ".join(t["text"] for t in ep["dialogue"]))
            save_series_state({
                "active": True,
                "topic": ep["topic"],
                "pillar": ep["pillar"],
                "setting": ep["setting"],
                "role_a": ep["role_a"],
                "role_b": ep["role_b"],
                "gen_a": ep["gen_a"],
                "gen_b": ep["gen_b"],
                "voice_pair": ep["voice_pair"],
                "theme": ep["theme"],
                "caption_hook": ep["caption_hook"],
                "current_part": ep["part_num"],
                "total_parts": ep["total_parts"],
                "history_scripts": history_scripts,
                "last_line": ep["dialogue"][-1]["text"],
            })
            print(f"Saved Part {ep['part_num']}/{ep['total_parts']} state for automatic continuation next run.")
        else:
            save_series_state({"active": False})
            print("Completed standalone/finale episode and cleared series_state.json.")

        time.sleep(90)
    finally:
        cloudinary.uploader.destroy(upload["public_id"], resource_type="video")
