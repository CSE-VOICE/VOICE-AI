# 🤖 VOICE-AI

## Proposal
This repository is dedicated to the development of the AI components for VOICE, a multimodal routine recommender designed to adapt to any circumstance.

Our core AI functionalities are structured into two key areas: training the VOICE model and extending its capabilities to the verbal domain.

## 1. Train VOICE Model
 Developing and training a generative VOICE AI language model designed to provide intelligent recommendations for smart home routines that seamlessly align with user needs, spanning from everyday scenarios to extraordinary and unique circumstances. This model leverages advanced contextual understanding and generative capabilities to deliver well-suited, personalized routine suggestions, ensuring adaptability and efficiency across a diverse range of user environments.

 To achieve this, we are fine-tuning a T5 model using transfer learning, leveraging a meticulously curated dataset of approximately 10,000 examples. This dataset was constructed by extracting diverse and comprehensive situational data from renowned large language models (LLMs) such as ChatGPT, Claude, Copilot, and Bard. By sourcing data from multiple LLMs, we ensured a broad spectrum of scenarios while carefully minimizing redundancies and maintaining the uniqueness of each entry. Each example pairs a specific situation with a corresponding smart home routine, enabling the model to learn complex relationships between context and actionable outcomes.

 This approach ensures that the VOICE AI language model is trained on a rich and diverse dataset, equipping it with the capability to generate highly accurate, context-aware, and user-specific routines for a wide range of smart home environments, thereby delivering an unparalleled level of adaptability and personalization.

## 2. Expanding VOICE to Verbal Domain
 Expanding VOICE’s capabilities to the verbal domain involves leveraging AI speaker technology to enable users to interact naturally by expressing their current situations through voice input. This process begins with converting spoken language into text, followed by a sophisticated analysis of the user’s speech to extract key attributes such as emotion, vocal tone, and speech pace. These attributes are meticulously tagged to enrich the converted text with contextual information.

 The enhanced text, now imbued with emotional and tonal tags, is then processed to generate coherent, context-aware sentences. Leveraging the ChatGPT API, these sentences are refined to ensure they accurately reflect the user’s emotions and circumstances while maintaining fluency and coherence.

 The final reconstructed text, enriched with emotional insights, serves as the input for our VOICE model, which is tasked with generating personalized smart home routines tailored to the user’s specific needs and environment. This end-to-end pipeline not only enhances the model’s ability to deliver nuanced and effective recommendations but also creates a seamless and emotionally intelligent interaction experience for the user.

