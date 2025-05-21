import streamlit as st
import openai
import os
import json
import numpy as np
import requests
import time
from datetime import datetime
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi
from googlesearch import search
from sentence_transformers import SentenceTransformer, util

# === Cấu hình OpenAI ===
client = openai.OpenAI()
st.set_page_config(page_title="GhostWriter AI", layout="wide")

STYLE_SAMPLE_DIR = "my_style_samples"
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

def build_reference_vectors():
    texts = st.session_state.sources
    if not texts:
        return
    embeddings = model_embed.encode(texts, convert_to_tensor=True)
    st.session_state.source_vectors = embeddings


# === SESSION STATE ===
if "hook" not in st.session_state:
    st.session_state.hook = ""
if "sections" not in st.session_state:
    st.session_state.sections = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "structure_titles" not in st.session_state:
    st.session_state.structure_titles = []
if "source_vectors" not in st.session_state:
    st.session_state.source_vectors = []

# === CÁC TONE PHONG CÁCH ĐỊNH SẴN ===
persona_tones = {
    "Giáo sư châm biếm": "witty, sarcastic, intellectually playful",
    "ASMR nhẹ nhàng": "calm, sensory, soothing and immersive",
    "Hài hước phóng đại": "goofy, absurd, highly exaggerated and comedic",
    "Kể như bạn thân": "casual, relatable, conversational",
    "Kinh dị thì thầm": "dark, suspenseful, vivid and unsettling",
    "Sử chuẩn giáo sư": "formal, precise, informative",
    "Triết lý huyền bí": "poetic, reflective, symbolic and deep"
}

# === STYLE VECTOR ===
def list_available_style_vectors():
    return [f for f in os.listdir() if f.startswith("my_style_vector") and f.endswith(".json")]

def load_style_vector_from_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return np.array(json.load(f))

# === TMProxy cache hỗ trợ ===
tmproxy_api_key = st.secrets["TM_PROXY_API_KEY"]
def get_tmproxy_with_cache(api_key):

    if "tmproxy" not in st.session_state:
        st.session_state.tmproxy = {}

    cache = st.session_state.tmproxy
    now = time.time()

    # Nếu còn hiệu lực thì dùng lại
    if cache and now < cache.get("expires_at", 0) - 30:
        return cache["proxy_url"]

    try:
        url = "https://tmproxy.com/api/proxy/get-new-proxy"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {"api_key": api_key, "id_location": 0, "id_isp": 0}

        res = requests.post(url, headers=headers, json=data, timeout=10)
        res.raise_for_status()
        res_json = res.json()

        # Nếu thành công → lưu và trả proxy mới
        if res_json["code"] == 0:
            proxy = res_json["data"]
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['https']}"
            expires_at = now + proxy["timeout"]

            st.session_state.tmproxy = {
                "proxy_url": proxy_url,
                "expires_at": expires_at
            }

            return proxy_url

        # Nếu bị giới hạn thời gian → dùng lại proxy cũ nếu có
        elif "retry after" in res_json.get("message", "").lower():
            if cache.get("proxy_url"):
                wait_msg = res_json["message"]
                st.info(f"🔁 {wait_msg}. Đang dùng lại proxy cũ.")
                return cache["proxy_url"]
            else:
                raise Exception(f"TMProxy chưa có proxy trước đó để dùng lại.")

        else:
            raise Exception(f"TMProxy Error: {res_json.get('message')}")

    except Exception as e:
        st.error(f"❌ Không thể lấy proxy từ TMProxy: {e}")
        return None

# === GIAO DIỆN CHÍNH ===
st.title("📝 GhostWriter AI")
topic = st.text_input("🎯 Nhập chủ đề hoặc nội dung video:")
pov_choice = st.selectbox("👤 Chọn ngôi kể:", ["first", "second", "third"], index=1)

selected_personas = st.multiselect("🎭 Chọn các phong cách hành văn muốn kết hợp:", list(persona_tones.keys()))
style_tone_instruction = ", ".join([persona_tones[p] for p in selected_personas]) if selected_personas else ""
strong_tone_prompt = ""
if style_tone_instruction:
    strong_tone_prompt = (
        f"Write this section in a distinctly {style_tone_instruction} tone.\n"
        f"Channel the spirit of a narrator who embodies these traits in full.\n"
        f"Avoid solemn or overly poetic language unless it enhances the comedic or satirical effect."
    )

# === NGUỒN THAM KHẢO GOOGLE ===
if st.button("🔎 Tìm link Google"):
    with st.spinner("Đang tìm kiếm trên Google..."):
        try:
            results = list(search(topic, num_results=25))
            st.session_state.search_links = results
        except Exception as e:
            st.error(f"Lỗi Google Search: {e}")

if "search_links" in st.session_state:
    selected_links = st.multiselect("Chọn link để trích nội dung:", st.session_state.search_links)

    if st.button("📄 Trích nội dung từ link đã chọn"):
        try:
            proxy_url = get_tmproxy_with_cache(tmproxy_api_key)

            if proxy_url:
                proxy_dict = {"http": proxy_url, "https": proxy_url}

                # Làm sạch trước khi thêm mới
                st.session_state.sources = []

                for link in selected_links:
                    try:
                        response = requests.get(link, proxies=proxy_dict, timeout=10)
                        text = trafilatura.extract(response.text)
                        if text:
                            st.session_state.sources.append(f"[SOURCE: {link}]\n{text.strip()}")
                        else:
                            st.warning(f"⚠️ Không trích xuất được nội dung từ: {link}")
                    except requests.exceptions.RequestException as e:
                        st.warning(f"⚠️ Lỗi khi truy cập {link}: {e}")

                build_reference_vectors()
                st.success("✅ Đã tạo vector từ nguồn tham khảo!")

            else:
                st.warning("⚠️ Không có proxy hợp lệ. Dừng tiến trình.")
        except Exception as e:
            st.error(f"❌ Lỗi proxy khi trích Google: {e}")


# === LẤY CAPTION YOUTUBE ===
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript

yt_url = st.text_input("Link YouTube")

if st.button("🎬 Lấy caption") and yt_url:
    try:
        video_id = parse_qs(urlparse(yt_url).query).get("v", [""])[0]
        proxy_url = get_tmproxy_with_cache(tmproxy_api_key)

        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
            transcript = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
            full_text = " ".join([x['text'] for x in transcript])

            st.session_state.sources.append(f"[YOUTUBE]\n{full_text.strip()}")
            build_reference_vectors()
            st.success("✅ Đã lấy caption từ YouTube qua TMProxy!")
        else:
            st.warning("⚠️ Không có proxy hợp lệ. Dừng tiến trình.")

    except CouldNotRetrieveTranscript:
        st.error("❌ Video không hỗ trợ phụ đề hoặc không thể lấy caption.")
    except Exception as e:
        st.error(f"❌ Lỗi lấy caption: {e}")


# === BUILD VECTOR ===
def build_reference_vectors():
    chunks = []
    for src in st.session_state.sources:
        # Bỏ dòng đầu nếu chứa metadata
        if src.startswith("[SOURCE:") or src.startswith("[YOUTUBE]"):
            src = "\n".join(src.split("\n")[1:])

        paragraphs = src.split("\n")
        for p in paragraphs:
            p = p.strip()
            if len(p) > 100:
                chunks.append(p)

    st.session_state.source_vectors = [
        (text, model_embed.encode(text, convert_to_tensor=False)) for text in chunks
    ]

def select_relevant_sources(title, top_k=1):
    title_vec = model_embed.encode(title, convert_to_tensor=False)
    scores = [(text, float(util.cos_sim(title_vec, vec))) for text, vec in st.session_state.source_vectors]
    scores.sort(key=lambda x: -x[1])
    return "\n\n".join([s[0] for s in scores[:top_k]])


# === TẠO HOOK ===
st.markdown("---")
st.subheader("✨ Viết Hook mở đầu video")

if st.button("🧠 Tạo Hook mở đầu"):
    with st.spinner("Đang tạo hook..."):

        # ✅ An toàn hơn khi check session_state
        if not st.session_state.get("source_vectors"):
            build_reference_vectors()

        refs = select_relevant_sources(topic, top_k=3)

        prompt = f"""
You are a creative ASMR-style content writer. Write a vivid, immersive, slightly witty YouTube hook intro for:
"{topic}"

References:
{refs}

Follow this format:
- Open with "Hey guys, tonight we..."
- Use vivid sensory imagery
- Include a funny warning: "you probably won't survive this..."
- End with a cozy CTA to like & relax
"""

        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            st.session_state.hook = res.choices[0].message.content.strip()

        except Exception as e:
            st.error(f"❌ GPT Error khi tạo hook: {e}")

st.text_area("✨ Hook mở đầu:", st.session_state.hook, height=150, key="hook_textarea")


# === GỢI Ý CẤU TRÚC NỘI DUNG ===
st.markdown("---")
st.subheader("📚 Gợi ý cấu trúc nội dung")
num_sections = st.slider("📑 Số lượng section mong muốn:", min_value=3, max_value=10, value=6, step=1)

if st.button("⚙️ Gợi ý lại cấu trúc"):
    with st.spinner("Đang sinh cấu trúc đề xuất..."):

        # Đảm bảo đã build vector trước khi chọn nguồn
        if "source_vectors" not in st.session_state or not st.session_state.source_vectors:
            build_reference_vectors()

        refs = select_relevant_sources(topic, top_k=3)  # chọn 3 đoạn tham khảo sát topic
        raw = ""  # tránh lỗi nếu parsing thất bại

        prompt = f"""
You are a YouTube script planner. Based on the topic below, generate a hook and exactly {num_sections} section titles in order.

Only return pure JSON. No explanation. Format:
{{
  "hook": "...",
  "sections": ["...", "...", "..."]
}}

Topic: {topic}
References:
{refs}
"""

        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=1000
            )
            raw = res.choices[0].message.content.strip()

            # Gỡ bỏ markdown nếu có
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)

            st.session_state.hook = parsed.get("hook", "")
            titles = parsed.get("sections", [])
            st.session_state.structure_titles = [(title, "") for title in titles]  # giữ đúng định dạng

        except Exception as e:
            st.error("❌ GPT trả về không đúng JSON hoặc lỗi khi phân tích:")
            st.code(raw if raw else "Không có phản hồi hợp lệ từ GPT.")


# === SECTION TIÊU ĐỀ ===
if st.session_state.structure_titles:
    st.markdown("#### 🧱 Tiêu đề các section (có thể chỉnh sửa hoặc xoá):")

    for i, (title, prompt) in enumerate(st.session_state.structure_titles):
        col1, col2 = st.columns([5, 1])

        with col1:
            new_title = st.text_input(f"Section {i+1} Title", title, key=f"title_{i}")
            st.session_state.structure_titles[i] = (new_title, prompt)  # cập nhật title, giữ nguyên prompt

        with col2:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.structure_titles.pop(i)
                st.experimental_rerun()


# === VIẾT NỘI DUNG MỖI SECTION ===
st.markdown("---")
st.subheader("✍️ Viết nội dung từng Section")

style_files = list_available_style_vectors()
selected_style_file = st.selectbox("🎨 Chọn phong cách cá nhân:", style_files if style_files else ["Không tìm thấy vector"])
word_count = st.slider("🔤 Số lượng từ mỗi section:", 100, 3000, 500, step=100)

# Hàm fallback nếu thiếu
def get_style_examples(vector, sample_count=3):
    try:
        if isinstance(vector, (list, np.ndarray)):
            return vector[:sample_count]
    except:
        return []
    return []

# Nếu có danh sách section cần viết
if st.session_state.structure_titles:
    for idx, (title, prompt) in enumerate(st.session_state.structure_titles):
        if st.button(f"✍️ Viết Section {idx+1}: {title}", key=f"gen_{idx}"):
            try:
                with st.spinner("Đang viết section..."):
                    vector = load_style_vector_from_file(selected_style_file)
                    references = select_relevant_sources(prompt)

                    # Lấy section trước đó nếu có
                    prev_title = st.session_state.sections[idx - 1]["title"] if idx > 0 and idx <= len(st.session_state.sections) else None
                    prev_content = st.session_state.sections[idx - 1]["user_edit"] if idx > 0 and idx <= len(st.session_state.sections) else None

                    examples = get_style_examples(vector)

                    # Tạo prompt đầy đủ
                    prompt_parts = [
                        "You are a writing assistant trained to match this personal narrative style.",
                        "Here are example paragraphs:\n" + "\n\n".join(examples),
                        "------------------------",
                        f"Hook: {st.session_state.hook}",
                        f"Section Title: {title}",
                        f"Section Prompt: {prompt}",
                    ]

                    if prev_title and prev_content:
                        prompt_parts.append("You are continuing a multi-part narrative script.")
                        prompt_parts.append(f"Previous Section Title: {prev_title}")
                        prompt_parts.append(f"Previous Section Content: {prev_content[:600]}")

                    prompt_parts.append(f"References:\n{references}")

                    # Bổ sung giọng điệu nếu có
                    if 'strong_tone_prompt' in globals():
                        prompt_parts.append(strong_tone_prompt)
                    else:
                        prompt_parts.append("Maintain consistent tone with the provided examples.")

                    if 'pov_choice' in globals():
                        prompt_parts.append(
                            f"Write a vivid, immersive section of about {word_count} words. "
                            f"Use {pov_choice} person point of view. "
                            f"Do NOT include the title in the content."
                        )
                    else:
                        prompt_parts.append(
                            f"Write a vivid, immersive section of about {word_count} words in the same tone. "
                            f"Do NOT include the title in the content."
                        )

                    # Gửi yêu cầu đến GPT
                    full_prompt = "\n\n".join(prompt_parts)
                    res = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.7,
                        max_tokens=min(int(word_count * 1.5), 6000)
                    )

                    output = res.choices[0].message.content.strip()

                    if idx < len(st.session_state.sections):
                        st.session_state.sections[idx]["user_edit"] = output
                        st.session_state.sections[idx]["title"] = title
                    else:
                        st.session_state.sections.append({
                            "title": title,
                            "user_edit": output
                        })

            except Exception as e:
                st.error(f"❌ GPT Error: {e}")


# === HIỂN THỊ SECTION ===
for i, sec in enumerate(st.session_state.sections):
    st.markdown(f"### 📦 {sec['title']}")
    sec["user_edit"] = st.text_area(f"Section {i+1} - Có thể chỉnh sửa tại đây:", sec["user_edit"], height=250, key=f"edit_{i}")

# === XUẤT TOÀN BỘ ===
st.markdown("---")
if st.session_state.sections:
    full_text = st.session_state.hook + "\n\n"
    for sec in st.session_state.sections:
        full_text += f"{sec['title']}\n{sec['user_edit']}\n\n"

    filename = f"ghostwriter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    st.download_button(
        label="📥 Tải toàn bộ nội dung (.txt)",
        data=full_text,
        file_name=filename,
        mime="text/plain"
    )

