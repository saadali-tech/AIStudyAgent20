import streamlit as st
import PyPDF2
import json
import re
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Study Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .hero {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e6e9ef;
        margin-bottom: 15px;
    }

    .difficulty {
        padding: 6px 12px;
        border-radius: 20px;
        background-color: #eef2ff;
        color: #4f46e5;
        font-weight: bold;
    }

    .weak {
        background-color: #fff1f2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
    }

    .strong {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "study_plan" not in st.session_state:
    st.session_state.study_plan = ""

if "teaching_content" not in st.session_state:
    st.session_state.teaching_content = ""

if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []

if "weak_topics" not in st.session_state:
    st.session_state.weak_topics = []


# ============================================================
# GROQ CLIENT
# ============================================================

def get_client():
    api_key = st.session_state.get("api_key")

    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", "")

    if not api_key:
        return None

    return Groq(api_key=api_key)



# ============================================================
# LLM FUNCTION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def ask_ai(prompt, temperature=0.3):

    client = get_client()

    if client is None:
        st.error(
            "Groq API key not found. Add it in the sidebar "
            "or Streamlit secrets."
        )
        return None

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert AI tutor and study assistant.

Your job is to help students understand lecture material,
prepare for exams, identify weak areas, and create effective
study plans.

Always prioritize information contained in the supplied
lecture material.

Do not invent facts that are not supported by the material
unless the user explicitly asks for additional explanation.

Explain difficult concepts clearly and progressively.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq AI error: {e}")
        return None



# ============================================================
# PDF EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def extract_pdf_text(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# CLEAN TEXT
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SUMMARIZE
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def generate_summary(text):

    prompt = f"""
Analyze the following lecture material.

Create an exam-focused summary.

Include:

1. Main topic
2. Key concepts
3. Important definitions
4. Important formulas or processes
5. Examples
6. Things students commonly confuse
7. Five key takeaways

Lecture material:

{text[:30000]}
"""

    return ask_ai(prompt)


# ============================================================
# FLASHCARDS
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def generate_flashcards(text):

    prompt = f"""
Create 10 useful study flashcards from this lecture.

Return ONLY valid JSON.

Format:

[
    {{
        "question": "Question",
        "answer": "Answer"
    }}
]

Lecture:

{text[:30000]}
"""

    result = ask_ai(prompt)

    if not result:
        return []

    try:

        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        return json.loads(result)

    except Exception:

        return []


# ============================================================
# MCQs
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def generate_mcqs(text, difficulty="Medium"):

    prompt = f"""
Create 10 multiple-choice questions from the lecture.

Difficulty: {difficulty}

Return ONLY valid JSON.

Format:

[
    {{
        "question": "Question",
        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],
        "answer": "Option A",
        "explanation": "Explanation"
    }}
]

Make the incorrect options plausible.

Lecture:

{text[:30000]}
"""

    result = ask_ai(prompt)

    if not result:
        return []

    try:

        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        return json.loads(result)

    except Exception:

        return []


# ============================================================
# TEACH LIKE I'M 12
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def teach_like_12(text, topic, difficulty):

    prompt = f"""
You are teaching a student.

Topic:
{topic}

Difficulty level:
{difficulty}

Explain the topic using the lecture material.

Start with a very simple explanation as if the student
were 12 years old.

Use:

- Simple language
- Everyday analogies
- Small examples
- Step-by-step explanations

Then gradually increase the complexity.

Structure:

LEVEL 1 — Like I'm 12
LEVEL 2 — High School
LEVEL 3 — University
LEVEL 4 — Exam Level

At each level explain the same concept more deeply.

Lecture:

{text[:30000]}
"""

    return ask_ai(prompt)


# ============================================================
# STUDY PLAN
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def generate_study_plan(text, days):

    prompt = f"""
Create a {days}-day personalized study plan based on this lecture.

The plan should include:

- Topics to study
- Daily goals
- Recommended study time
- Practice activities
- Revision sessions
- Self-testing
- Final review

Return a clear day-by-day plan.

Lecture:

{text[:30000]}
"""

    return ask_ai(prompt)


# ============================================================
# WEAK TOPIC DETECTION
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10)
def analyze_weak_topics(results):

    if not results:
        return []

    result_text = json.dumps(results)

    prompt = f"""
Analyze these quiz results.

Identify:

1. Topics the student understands
2. Topics the student struggles with
3. Likely misconceptions
4. Recommended topics for revision

Quiz results:

{result_text}

Return JSON:

{{
    "strong_topics": [],
    "weak_topics": [],
    "recommendations": []
}}
"""

    result = ask_ai(prompt)

    if not result:
        return []

    try:

        result = result.replace("```json", "")
        result = result.replace("```", "")

        return json.loads(result)

    except Exception:

        return []


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<h1>🧠 AI Study Agent</h1>

<p>
Turn your lecture PDFs into an intelligent,
personalized study system.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your key is kept in this session."
    )

    if api_key:
        st.session_state.api_key = api_key

    st.divider()

    st.subheader("📚 Study Settings")

    study_days = st.slider(
        "Study plan duration",
        min_value=1,
        max_value=30,
        value=7
    )

    difficulty = st.select_slider(
        "Question difficulty",
        options=[
            "Easy",
            "Medium",
            "Hard",
            "Expert"
        ],
        value="Medium"
    )

    st.divider()

    st.caption(
        "AI Study Agent • Built with Python + Streamlit"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload Lecture Material")

uploaded_files = st.file_uploader(
    "Upload one or more lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    combined_text = ""

    for uploaded_file in uploaded_files:

        with st.spinner(
            f"Reading {uploaded_file.name}..."
        ):

            pdf_text = extract_pdf_text(uploaded_file)

            combined_text += (
                f"\n\n--- {uploaded_file.name} ---\n\n"
                + pdf_text
            )

    st.session_state.pdf_text = clean_text(combined_text)

    st.success(
        f"Successfully processed {len(uploaded_files)} PDF(s)."
    )

    word_count = len(
        st.session_state.pdf_text.split()
    )

    st.info(
        f"📚 Extracted approximately {word_count:,} words."
    )


# ============================================================
# STOP IF NO PDF
# ============================================================

if not st.session_state.pdf_text:

    st.info(
        "👆 Upload your lecture PDFs to start studying."
    )

    st.markdown("""
### What you will get

- 📄 Intelligent lecture summaries
- 🧠 AI-generated flashcards
- 📝 Exam-style MCQs
- 🎯 Personalized quizzes
- 📊 Weak-topic detection
- 📅 Personalized study plans
- 👨‍🏫 Progressive AI teaching
- 💬 Ask questions about your lecture
"""
)

    st.stop()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📚 Overview",
    "🧠 Flashcards",
    "📝 MCQs",
    "🎯 Quiz",
    "📊 Weak Topics",
    "📅 Study Plan",
    "👨‍🏫 Teach Me",
    "💬 Ask AI"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header("📚 Lecture Overview")

    if st.button(
        "✨ Generate AI Summary",
        type="primary"
    ):

        with st.spinner("Analyzing your lecture..."):

            summary = generate_summary(
                st.session_state.pdf_text
            )

            st.session_state.summary = summary

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# ============================================================
# FLASHCARDS
# ============================================================

with tabs[1]:

    st.header("🧠 AI Flashcards")

    if st.button("Generate Flashcards"):

        with st.spinner(
            "Creating intelligent flashcards..."
        ):

            cards = generate_flashcards(
                st.session_state.pdf_text
            )

            st.session_state.flashcards = cards

    cards = st.session_state.flashcards

    if cards:

        for index, card in enumerate(cards):

            with st.expander(
                f"Card {index + 1}: {card.get('question', '')}"
            ):

                st.write(
                    card.get("answer", "")
                )

    else:

        st.info(
            "Click Generate Flashcards to create your deck."
        )


# ============================================================
# MCQs
# ============================================================

with tabs[2]:

    st.header("📝 AI Exam Questions")

    if st.button(
        "Generate MCQs",
        key="generate_mcqs"
    ):

        with st.spinner(
            "Generating exam-style questions..."
        ):

            mcqs = generate_mcqs(
                st.session_state.pdf_text,
                difficulty
            )

            st.session_state.mcqs = mcqs

    mcqs = st.session_state.mcqs

    if mcqs:

        for index, q in enumerate(mcqs):

            st.markdown(
                f"### Question {index + 1}"
            )

            st.write(q["question"])

            selected = st.radio(
                "Choose an answer:",
                q["options"],
                key=f"mcq_{index}"
            )

            if st.button(
                "Check Answer",
                key=f"check_{index}"
            ):

                if selected == q["answer"]:

                    st.success(
                        "✅ Correct!"
                    )

                else:

                    st.error(
                        f"❌ Correct answer: {q['answer']}"
                    )

                st.info(
                    q["explanation"]
                )

            st.divider()


# ============================================================
# PERSONALIZED QUIZ
# ============================================================

with tabs[3]:

    st.header("🎯 Personalized Quiz")

    st.write(
        "Take a quiz and let the AI discover your weak areas."
    )

    if st.button(
        "Start Personalized Quiz"
    ):

        with st.spinner(
            "Preparing your personalized quiz..."
        ):

            quiz = generate_mcqs(
                st.session_state.pdf_text,
                difficulty
            )

            st.session_state.quiz_results = []

            for i, question in enumerate(quiz):

                st.markdown(
                    f"### {i + 1}. {question['question']}"
                )

                answer = st.radio(
                    "Your answer:",
                    question["options"],
                    key=f"personal_{i}"
                )

                st.session_state.quiz_results.append({
                    "question": question["question"],
                    "selected": answer,
                    "correct": question["answer"],
                    "explanation": question["explanation"]
                })

            if st.button(
                "Submit Quiz",
                key="submit_personalized"
            ):

                score = 0

                for result in st.session_state.quiz_results:

                    if result["selected"] == result["correct"]:

                        score += 1

                st.success(
                    f"Your score: {score}/{len(st.session_state.quiz_results)}"
                )

                st.session_state.weak_topics = (
                    analyze_weak_topics(
                        st.session_state.quiz_results
                    )
                )


# ============================================================
# WEAK TOPICS
# ============================================================

with tabs[4]:

    st.header("📊 Weak Topic Detection")

    weak = st.session_state.weak_topics

    if weak:

        strong_topics = weak.get(
            "strong_topics",
            []
        )

        weak_topics = weak.get(
            "weak_topics",
            []
        )

        recommendations = weak.get(
            "recommendations",
            []
        )

        st.subheader("💪 Strong Areas")

        for topic in strong_topics:

            st.markdown(
                f"""
                <div class="strong">
                ✅ {topic}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("⚠️ Weak Areas")

        for topic in weak_topics:

            st.markdown(
                f"""
                <div class="weak">
                ⚠️ {topic}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader(
            "🎯 AI Recommendations"
        )

        for recommendation in recommendations:

            st.write(
                f"• {recommendation}"
            )

    else:

        st.info(
            "Complete the personalized quiz first."
        )


# ============================================================
# STUDY PLAN
# ============================================================

with tabs[5]:

    st.header("📅 Personalized Study Plan")

    st.write(
        f"AI-generated {study_days}-day study roadmap."
    )

    if st.button(
        "Generate Study Plan"
    ):

        with st.spinner(
            "Creating your study roadmap..."
        ):

            plan = generate_study_plan(
                st.session_state.pdf_text,
                study_days
            )

            st.session_state.study_plan = plan

    if st.session_state.study_plan:

        st.markdown(
            st.session_state.study_plan
        )


# ============================================================
# TEACH ME
# ============================================================

with tabs[6]:

    st.header(
        "👨‍🏫 Teach Me Like I'm 12"
    )

    st.write(
        "Learn a difficult concept progressively."
    )

    topic = st.text_input(
        "What topic do you want to learn?",
        placeholder="Example: Photosynthesis"
    )

    teaching_level = st.select_slider(
        "Starting difficulty",
        options=[
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        value="Beginner"
    )

    if st.button(
        "🚀 Teach Me",
        type="primary"
    ):

        if not topic:

            st.warning(
                "Enter a topic first."
            )

        else:

            with st.spinner(
                "Your AI tutor is preparing the lesson..."
            ):

                teaching = teach_like_12(
                    st.session_state.pdf_text,
                    topic,
                    teaching_level
                )

                st.session_state.teaching_content = teaching

    if st.session_state.teaching_content:

        st.markdown(
            st.session_state.teaching_content
        )


# = ============================================================
# ASK AI
# ============================================================

with tabs[7]:

    st.header("💬 Ask Your Lecture")

    question = st.text_area(
        "Ask anything about the uploaded lecture",
        placeholder=(
            "Explain the most important concept "
            "from this lecture..."
        )
    )

    if st.button(
        "Ask AI",
        type="primary"
    ):

        if not question:

            st.warning(
                "Enter a question."
            )

        else:

            prompt = f"""
Answer the student's question using the lecture.

Student question:

{question}

Lecture:

{st.session_state.pdf_text[:30000]}

Rules:

- Explain clearly.
- Use examples when useful.
- Mention when something is not covered
  in the lecture.
- Do not invent lecture-specific information.
"""

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_ai(
                    prompt,
                    temperature=0.2
                )

            if answer:

                st.markdown(
                    "### 🤖 AI Tutor"
                )

                st.markdown(answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 AI Study Agent | Python + Streamlit | Powered by Groq"
)
