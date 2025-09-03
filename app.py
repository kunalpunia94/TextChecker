import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Answer Comparison App")

# Inputs
correct_answer = st.text_area("Enter the Correct Answer (Reference)")
student_answer = st.text_area("Enter the Student Answer")

if st.button("Compare Answers"):
    if correct_answer.strip() and student_answer.strip():
        # Step 1: Embedding similarity
        emb_correct = embedder.encode([correct_answer])[0]
        emb_student = embedder.encode([student_answer])[0]

        cosine_sim = np.dot(emb_correct, emb_student) / (norm(emb_correct) * norm(emb_student))

        # Step 2: LLM Prompt
        prompt = f"""
        Compare the student's answer with the correct answer semantically.
        
        Correct Answer:
        {correct_answer}
        
        Student Answer:
        {student_answer}
        
        Cosine Similarity Score: {cosine_sim:.2f}
        
        Instructions:
        - Always respond in this structured format:
          Classification: <Similar / Partially Similar / Different>
          Reason: <short explanation of why they are classified this way>
          Sources:
            - Correct Answer: <show correct answer>
            - Student Answer: <show student answer>
          Expected Student Correction:
            - If classification is "Similar", respond: "Not needed – already correct."
            - If classification is "Partially Similar" or "Different", do NOT repeat the full correct answer. 
              Instead, provide only the missing or incorrect part that should be added/changed in the student's answer.
            - If unsure because the board’s response does not match column F or cannot be verified with methodology, respond: "Uncertain – cannot verify against methodology."
        """




        # Step 3: Call Groq LLM
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        result = response.choices[0].message.content

        # Display results
        st.subheader("Evaluation Result:")
        st.write(result)

        st.subheader("Similarity Score:")
        st.metric(label="Cosine Similarity", value=f"{cosine_sim:.2f}")


