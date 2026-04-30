"""
BharatLLM Chat Interface
=========================
Interactive terminal chat with your trained BharatLLM.
Supports all Indian languages + English + Hinglish.

Usage:
    python scripts/chat.py --model ./outputs/bharat-llm-7b
    python scripts/chat.py --model ./outputs/bharat-llm-7b --lang hi
    python scripts/chat.py --model ./outputs/bharat-llm-7b --quantize 4bit
"""

import sys
import argparse
from pathlib import Path

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🇮🇳  BharatLLM — India's Open Language Model                   ║
║                                                                  ║
║   Built by merging open-weight models + Indian language data     ║
║   Supports: हिंदी • বাংলা • தமிழ் • తెలుగు • मराठी • English    ║
║                                                                  ║
║   Commands:                                                      ║
║   /lang hi      — Switch to Hindi                               ║
║   /lang en      — Switch to English                             ║
║   /lang ta      — Switch to Tamil                               ║
║   /system <msg> — Set system prompt                             ║
║   /clear        — Clear conversation history                    ║
║   /save         — Save conversation to file                     ║
║   /quit         — Exit                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

SYSTEM_PROMPTS = {
    "hi": "आप एक सहायक AI हैं जो भारतीय भाषाओं में विशेष रूप से हिंदी में बात कर सकते हैं। आप भारतीय संस्कृति, इतिहास और जीवनशैली के बारे में गहरी जानकारी रखते हैं।",
    "bn": "আপনি একজন সহায়ক AI যিনি বাংলা সহ ভারতীয় ভাষায় কথা বলতে পারেন। আপনি ভারতীয় সংস্কৃতি এবং ইতিহাস সম্পর্কে গভীর জ্ঞান রাখেন।",
    "ta": "நீங்கள் தமிழ் உட்பட இந்திய மொழிகளில் பேசக்கூடிய ஒரு உதவி AI ஆவீர்கள். இந்திய கலாச்சாரம் மற்றும் வரலாறு பற்றி ஆழமான அறிவு கொண்டவர்.",
    "en": "You are BharatLLM, a helpful AI assistant specialized in Indian languages, culture, and knowledge. You can seamlessly switch between English, Hindi, Bengali, Tamil, and other Indian languages based on the user's preference.",
    "hinglish": "Aap ek helpful AI assistant hain jo Indian languages mein specialization rakhte hain. Aap Hinglish (Hindi-English mix) mein bhi baat kar sakte hain.",
}


class BharatLLMChat:
    """Interactive chat interface for BharatLLM."""

    def __init__(
        self,
        model_path: str,
        lang: str = "en",
        quantize: str = "4bit",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_path = model_path
        self.lang = lang
        self.quantize = quantize
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.conversation_history = []
        self.system_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])

    def load_model(self):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"\nLoading BharatLLM from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )

        if self.quantize == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif self.quantize == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Full precision
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

        self.model.eval()
        print("✅ Model loaded successfully!\n")

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"   Parameters: {total_params:.1f}B")
        print(f"   Quantization: {self.quantize}")
        print(f"   Language mode: {self.lang}")

        return self

    def format_prompt(self, user_message: str) -> str:
        """Format the full conversation as a prompt."""
        # Build conversation context
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"

        for turn in self.conversation_history:
            prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"

        return prompt

    def generate(self, user_message: str) -> str:
        """Generate response for user message."""
        import torch

        prompt = self.format_prompt(user_message)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.max_new_tokens,
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.strip()

        # Remove any trailing tags
        for tag in ["<|im_end|>", "<|im_start|>", "</s>"]:
            response = response.replace(tag, "").strip()

        # Save to history
        self.conversation_history.append({
            "user": user_message,
            "assistant": response,
        })

        # Keep last 10 turns to avoid context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response

    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if should continue."""
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
            print("\nAlvida! / अलविदा! / Good Bye! 🙏")
            return False

        elif cmd == "/lang":
            if args in SYSTEM_PROMPTS:
                self.lang = args
                self.system_prompt = SYSTEM_PROMPTS[args]
                lang_names = {"hi": "Hindi हिंदी", "bn": "Bengali বাংলা",
                             "ta": "Tamil தமிழ்", "en": "English",
                             "hinglish": "Hinglish"}
                print(f"  Switched to {lang_names.get(args, args)} mode")
            else:
                print(f"  Unknown language. Supported: {list(SYSTEM_PROMPTS.keys())}")

        elif cmd == "/system":
            if args:
                self.system_prompt = args
                print(f"  System prompt updated")
            else:
                print(f"  Current system prompt:\n  {self.system_prompt}")

        elif cmd == "/clear":
            self.conversation_history = []
            print("  Conversation cleared")

        elif cmd == "/save":
            import json, datetime
            filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "lang": self.lang,
                    "system_prompt": self.system_prompt,
                    "history": self.conversation_history,
                }, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {filename}")

        elif cmd == "/temp":
            try:
                self.temperature = float(args)
                print(f"  Temperature set to {self.temperature}")
            except ValueError:
                print("  Usage: /temp 0.7")

        elif cmd == "/help":
            print(WELCOME_BANNER)

        return True

    def run(self):
        """Main chat loop."""
        print(WELCOME_BANNER)

        while True:
            try:
                # Language-specific prompts
                prompts = {
                    "hi": "आप: ", "bn": "আপনি: ", "ta": "நீங்கள்: ",
                    "en": "You: ", "hinglish": "Aap: ",
                }
                prompt_str = prompts.get(self.lang, "You: ")

                user_input = input(f"\n{prompt_str}").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Generate response
                print("\nBharatLLM: ", end="", flush=True)
                response = self.generate(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\n(Use /quit to exit)")
            except EOFError:
                break


def main():
    parser = argparse.ArgumentParser(description="BharatLLM Interactive Chat")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to BharatLLM model directory")
    parser.add_argument("--lang", type=str, default="en",
                        choices=["hi", "bn", "ta", "te", "mr", "gu", "en", "hinglish"],
                        help="Default language mode")
    parser.add_argument("--quantize", type=str, default="4bit",
                        choices=["4bit", "8bit", "none"],
                        help="Quantization level (4bit fits in 6GB VRAM!)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    chat = BharatLLMChat(
        model_path=args.model,
        lang=args.lang,
        quantize=args.quantize,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    chat.load_model()
    chat.run()


if __name__ == "__main__":
    main()
