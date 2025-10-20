import streamlit as st
import requests
import json

TOGETHER_API_KEY = "tgp_v1_E6vk_inHoSbaD7_ebgzTI_bkB_bZq5fmHReCKGYAoA0"
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

def detect_module_suggestion(user_input):
    suggestions = []
    text = user_input.lower()
    if any(k in text for k in ["irregular period", "facial hair", "acne", "weight gain", "pcos"]):
        suggestions.append("🌼 You may want to check the **PCOS Risk Detector**.")
    if any(k in text for k in ["pain during sex", "pelvic pain", "heavy bleeding", "endometriosis"]):
        suggestions.append("🌸 Your symptoms may relate to the **Endometriosis Risk Detector**.")
    if any(k in text for k in ["no eye contact", "not speaking", "delayed talking", "autism", "toddler"]):
        suggestions.append("👶 You might find the **Autism Risk Detector (for toddlers)** helpful.")
    return suggestions

def main():
    st.set_page_config(page_title="SympSense Assistant", page_icon="🪄")
    st.title("🪄 SympSense Assistant")
    st.markdown("I'm here to listen to your symptoms and guide you toward helpful tools or support.")

    st.header("💬 Tell me what's been bothering you:")
    st.markdown("**Describe any symptoms, feelings, or concerns:**")
    user_input = st.text_area(" ", height=150)

    if st.button("Ask Assistant") and user_input.strip():
        with st.spinner("Thinking..."):

            # Step 1: Suggestions First
            suggestions = detect_module_suggestion(user_input)
            for s in suggestions:
                st.info(s)

            # Step 2: Fetch streamed response
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a caring assistant for the SympSense app, which helps users assess risk for PCOS, Endometriosis, and Autism in toddlers. "
                            "Offer short, helpful advice about what they can do right now to feel better or get relief. "
                            "End with supportive words, but don’t repeat that you're not a doctor or overuse generic warnings."
                        )
                    },
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.7,
                "stream": True
            }

            st.markdown("### 🤖 Assistant's Response")
            full_reply = ""
            placeholder = st.empty()

            try:
                with requests.post(TOGETHER_API_URL, headers=headers, json=payload, stream=True) as response:
                    for line in response.iter_lines():
                        try:
                            if line and line.startswith(b"data: "):
                                line_data = json.loads(line[6:])
                                delta = line_data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    full_reply += content
                                    placeholder.markdown(full_reply)
                        except Exception:
                            continue  # skip lines that aren't JSON formatted

            except Exception as e:
                st.error(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
