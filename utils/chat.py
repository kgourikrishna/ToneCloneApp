# utils/chat.py

from openai import OpenAI
import json

class ToneCloneChatAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the chat agent.
        """
        self.llm_model = "gpt-4o-mini"
        self.client = OpenAI(api_key=api_key)
        self.messages = []
        self.add_system_message()

        prompt = (
            "Here is a summary of effects possibly detected in a guitar recording."
            "The information comes from a machine learning model, and while often helpful, there will be false detections."
            "Please interpret this summary carefully. Focus on how each effect might be shaping the tone, using language "
            "to indicate uncertainty when effects appear inconsistently or with lower confidence."
            "These classifications come from an ML model, so except when you're very sure you should treat detections as possibly inaccurate."
            "Do not reference specific numbers, segment counts, or confidence values in your response."
            "Use natural language to explain what these effects are likely doing to the tone, and how the player might be using them together."
            "Do not use any formatting besides paragraphs in your response."
            "Do not include introduction or conclusion phrases in your response."
            "Include the phrase \"if present\" for any uncertain effects."
            "Respond with a JSON object that gives a description for each effect, "
            "a well-known song or artist that uses it, "
            "and specific pedal recommendations for recreating the sound."
        )

        lead = (
            "Here’s a breakdown of the effects that seem to be shaping your guitar tone. "
            "This analysis isn’t always perfect, but it should give you a helpful starting point for understanding the sounds in your recording and how to recreate them."
        )

    def add_system_message(self):
        """Adds a system message to set the conversation context when the conversation starts."""
        system_message = {
            "role": "system",
            "content": "You are a friendly and knowledgeable guitar effects tutor helping users understand how different effects shape guitar tone. "
                       "Your goal is to make complex tone concepts approachable, especially for beginners. "
                       "You explain how common effects influence the sound, and provide real-world references when helpful. "
                       "Use clear, conversational language, and avoid technical jargon unless it's explained. "
                       "You do not ask follow-up questions. "
                       "Focus on helping users feel more confident in exploring and recreating guitar sounds. "
        }
        self.messages.append(system_message)

    def send_message(self, user_message: str):
        """
        Sends a message to the ChatGPT agent and returns the response.
        user_message: The message to send to the agent.
        returns: The agent's response as a string.
        """
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=self.messages
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def reset_conversation(self):
        """Clears the conversation history."""
        self.messages = []

    def summarize_effects(self, effects: list[str], model_summary) -> dict:
        if not effects or not isinstance(effects, list):
            raise ValueError("Effects must be a non-empty list of strings.")

        if len(effects) > 3:
            effects = effects[:3]  # Limit to 3



        user_message ={ 
                        "role": "user",
                        "content": f"{model_summary}. "
                        "Respond with a JSON object that provides information about each of these effects. "
                        "Use a friendly, confident tone and assume the user is an aspiring guitarist eager to learn. "
                        "Each attribute in the JSON should be written in full, engaging sentences with no formatting. "
                        "For `detailed_description_and_common_use_cases`, write 3–5 sentences explaining how the effect works, what it sounds like, and when it's typically used. "
                        "For `extensive_pedal_recommendations_with_brief_justifications`, give 2–3 specific pedal models and explain briefly why each one is a good choice. "
                        "For `notable_artist_usage`, give the name of a famous artist or song that uses this effect, and explain briefly how the effect is used in that musical context."
                }
        
        print(user_message)
        
        self.messages.append(user_message)
        print(self.messages)

        response = self.client.responses.create(
            model=self.llm_model,
            input=self.messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "guitar_effect_summary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "effects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" },
                                        "confidence_judgement": {
                                            "type": "string",
                                            "enum": ["Uncertain", "Confident", "Very Confident"]
                                        },
                                        "detailed_description_and_common_use_cases": { "type": "string" },
                                        "notable_artist_usage": { "type": "string" },
                                        "extensive_pedal_recommendations_with_brief_justifications": { "type": "string" }
                                    },
                                    "required": ["name", "confidence_judgement", "detailed_description_and_common_use_cases", "notable_artist_usage", "extensive_pedal_recommendations_with_brief_justifications"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["effects"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )

        return json.loads(response.output_text)