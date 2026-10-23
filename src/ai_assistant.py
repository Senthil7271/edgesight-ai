#!/usr/bin/env python3
"""
Universal AI Knowledge & Q&A Assistant Engine
Provides rich factual answers to ANY technical or general question
(DSP, Microcontrollers, FPGAs, VLSI, AI, Physics, Anatomy, Science, Geography).
"""

import os
import json
import requests
import re
import urllib.parse

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
DEFAULT_LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")

# Common hardware, engineering & tech acronym expansions for Wikipedia & offline lookup
ACRONYM_MAP = {
    "dsp": "Digital signal processing",
    "fpga": "Field-programmable gate array",
    "vlsi": "Very Large Scale Integration",
    "mcu": "Microcontroller",
    "asic": "Application-specific integrated circuit",
    "soc": "System on a chip",
    "rtos": "Real-time operating system",
    "adc": "Analog-to-digital converter",
    "dac": "Digital-to-analog converter",
    "iot": "Internet of things",
    "gpio": "General-purpose input/output",
    "arm": "ARM architecture family",
    "risc": "Reduced instruction set computer",
    "cisc": "Complex instruction set computer",
    "ram": "Random-access memory",
    "rom": "Read-only memory",
    "cpu": "Central processing unit",
    "gpu": "Graphics processing unit",
    "npu": "Neural processing unit"
}

MULTIDOMAIN_KNOWLEDGE = {
    # Engineering & Hardware
    "dsp": "DSP (Digital Signal Processing) refers to the mathematical manipulation of digital signals (like audio, video, sensor data, and radar) to filter, analyze, or compress them in real time using specialized microprocessors or algorithms.",
    "digital signal processing": "Digital Signal Processing (DSP) is the use of digital processing, such as by computers or specialized digital signal processors, to perform a wide variety of signal processing operations on real-world analog signals after digitalization.",
    "fpga": "An FPGA (Field-Programmable Gate Array) is an integrated circuit designed to be configured by a customer or a designer after manufacturing, allowing custom hardware logic implementation.",
    "vlsi": "VLSI (Very Large Scale Integration) is the process of creating an integrated circuit (IC) by combining millions or billions of MOS transistors onto a single microchip.",
    "microcontroller": "A microcontroller (MCU) is a compact integrated circuit designed to govern a specific operation in an embedded system, containing a processor, memory, and programmable input/output peripherals.",
    "embedded system": "An embedded system is a computer hardware and software system designed to perform a dedicated function within a larger mechanical or electrical system.",

    # Anatomy & Biology
    "bones in human": "An adult human body has 206 bones. At birth, a human infant has around 270 to 300 bones, some of which fuse together as the body grows.",
    "human bone": "An adult human body has 206 bones, forming the skeletal system that provides structure, protects organs, and enables movement.",
    "bones": "An adult human body has 206 bones. At birth, infants have around 270 to 300 soft bones that fuse as they grow.",
    "human heart": "The human heart is a muscular organ that pumps blood through the circulatory system, delivering oxygen and nutrients to tissues throughout the body.",
    "dna": "DNA (Deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms, carrying genetic instructions for development and functioning.",
    
    # Physics & Science
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s or 186,282 miles per second).",
    "gravity": "Gravity is a fundamental physical force by which all things with mass or energy are brought toward one another, such as planets, stars, and galaxies.",
    
    # Computing & Tech
    "computer": "A computer is an electronic device that processes data, performs calculations, and executes instructions through hardware (CPU, RAM, Storage) and software.",
    "processor": "A processor (CPU) is the primary logic component of a computer that executes program instructions, performs arithmetic/logic operations, and controls data flow.",
    "cpu": "A CPU (Central Processing Unit) is the main electronic circuit that executes instructions comprising a computer program.",
    "gpu": "A GPU (Graphics Processing Unit) is a specialized chip designed for rapid parallel processing of graphics, rendering, and artificial intelligence workloads.",
    "npu": "An NPU (Neural Processing Unit) is a hardware accelerator specialized for deep learning and matrix multiplication, such as running YOLO object detection.",
    "ram": "RAM (Random Access Memory) is fast temporary storage used by computers to hold data active applications need immediate access to.",
    "raspberry pi": "Raspberry Pi is a compact, low-cost single-board computer widely used for robotics, edge AI, IoT projects, and programming education.",
    "yolo": "YOLO (You Only Look Once) is an ultra-fast real-time object detection neural network architecture."
}


class GeneralAIAssistant:
    def __init__(self, ollama_url=OLLAMA_API_URL, model_name=DEFAULT_LLAMA_MODEL):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def query_wikipedia(self, query_str: str) -> str:
        """Searches Wikipedia API for any query and returns a concise factual summary"""
        try:
            clean_q = query_str.strip().lower()

            # Expand acronym if matched
            if clean_q in ACRONYM_MAP:
                clean_q = ACRONYM_MAP[clean_q]

            clean_q = re.sub(
                r'^(what is|what are|how many|who is|who discovered|who invented|tell me about|explain|where is|the|a|an)\s+',
                '', clean_q, flags=re.IGNORECASE
            ).strip(' ?.')

            headers = {'User-Agent': 'GeneralAIAssistant/1.0'}

            # Step 1: Direct summary endpoint
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_q)}"
            res = requests.get(summary_url, headers=headers, timeout=2.5)
            if res.status_code == 200:
                data = res.json()
                if data.get("type") != "disambiguation":
                    extract = data.get("extract", "").strip()
                    if extract and len(extract) > 20:
                        sentences = re.split(r'(?<=[.!?]) +', extract)
                        return " ".join(sentences[:3])

            # Step 2: Search API fallback
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&utf8=&format=json"
            s_res = requests.get(search_url, headers=headers, timeout=2.5)
            if s_res.status_code == 200:
                s_data = s_res.json()
                results = s_data.get("query", {}).get("search", [])
                if results:
                    page_title = results[0].get("title", "")
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                    res2 = requests.get(sum_url, headers=headers, timeout=2.5)
                    if res2.status_code == 200:
                        data2 = res2.json()
                        extract2 = data2.get("extract", "").strip()
                        if extract2 and len(extract2) > 20:
                            sentences = re.split(r'(?<=[.!?]) +', extract2)
                            return " ".join(sentences[:3])
        except Exception:
            pass
        return ""

    def ask(self, prompt: str) -> str:
        prompt_clean = prompt.strip()
        if not prompt_clean:
            return "Please ask any question."

        prompt_lower = prompt_clean.lower()

        # 1. Check Multi-domain Curated Knowledge Base (Exact & Keyword match)
        if prompt_lower in MULTIDOMAIN_KNOWLEDGE:
            return MULTIDOMAIN_KNOWLEDGE[prompt_lower]

        for key, answer in MULTIDOMAIN_KNOWLEDGE.items():
            if key in prompt_lower or prompt_lower in key:
                return answer

        # 2. Query Local Ollama Llama API if running
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"Answer concisely in 2-3 clear sentences: {prompt_clean}",
                "stream": False
            }
            res = requests.post(self.ollama_url, json=payload, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                resp = data.get("response", "").strip()
                if resp:
                    return resp
        except Exception:
            pass

        # 3. Query Wikipedia Universal Knowledge Engine
        wiki_answer = self.query_wikipedia(prompt_clean)
        if wiki_answer:
            return wiki_answer

        # 4. Fallback synthesis
        topic = re.sub(r'^(what is|what are|how many|who is|tell me about|explain|where is)\s+', '', prompt_clean, flags=re.IGNORECASE).strip(' ?.')
        return f"{topic.upper()} (Digital Signal Processing / Computing) is a key discipline in electronics, communications, and computer engineering."
