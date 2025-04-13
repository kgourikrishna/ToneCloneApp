# utils/analyzer.py

import numpy as np
import torch
import librosa
import wave
import boto3
import json
import io
from collections import defaultdict, Counter
from utils.chat import ToneCloneChatAgent

class ToneCloneAnalyzer:
    
    # EFFECT_LABELS = {
    #     "ODV": "overdrive", "DST": "distortion", "FUZ": "fuzz", "TRM": "tremolo",
    #     "PHZ": "phaser", "FLG": "flanger", "CHR": "chorus", "DLY": "delay", "HLL": "hall_reverb",
    #     "PLT": "plate_reverb", "OCT": "octaver", "FLT": "auto_filter"
    # }
    # LABEL_NAMES = list(EFFECT_LABELS.values())  # Ordered list of effect names
    # NUM_CLASSES = len(LABEL_NAMES)

    def __init__(self, file_path, openai_key):
        self.file_path = file_path
        self.sample_length = 10
        self.overlap = 0
        self.target_sr = 32000
        self.n_fft = 2048
        self.hop_length = 512
        self.num_mels = 128
        
        #self.endpoint_name = 'ToneClone'
        self.endpoint_name = 'ToneClonePANN'
        self.max_segments_per_request = 16
        self.spectrograms = None
        self.raw_predictions = None
        self.thresholds = {
            "chorus": (0.5, 0.7, 0.9),
            "flanger": (0.5, 0.7, 0.9),
            "delay": (0.5, 0.7, 0.9),
            "plate reverb": (0.5, 0.7, 0.9),
            "overdrive": (0.01, 0.3, 0.5),
            "distortion": (0.01, 0.3, 0.5),
            "fuzz": (0.01, 0.3, 0.5),
            "tremolo": (0.5, 0.7, 0.9),
            "phaser": (0.5, 0.7, 0.9),
            "hall reverb": (0.5, 0.7, 0.9),
            "octaver": (0.9999, 0.999999, 0.99999999),
            "auto filter": (0.9999, 0.999999, 0.99999999)
        }
        self.families = {
            "distortion_family": ["overdrive", "distortion", "fuzz"],
            "modulation_family": ["chorus", "flanger", "phaser"],
            "reverb_family": ["plate reverb", "hall reverb"]
        }
        self.summarized_segment_results = None
        self.predictions_summary_for_llm = None
        self.top_effects = None
        self.effect_json = None
        self.chat_agent = ToneCloneChatAgent(openai_key) 

    def split_wav(self):
        if self.file_path is None:
            raise RuntimeError("No file path specified. Please set a file path before processing.")

        with wave.open(self.file_path, 'rb') as wav:
            num_channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            #print(f"Original sample rate: {sample_rate} Hz")
            num_frames = wav.getnframes()

            audio_data = np.frombuffer(wav.readframes(num_frames), dtype=np.int16)
            if num_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)  # Convert stereo to mono

            # Resample if needed
            if sample_rate != self.target_sr:
                #print(f"Resampling from {sample_rate} Hz to {self.target_sr} Hz...")
                audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=self.target_sr)
                sample_rate = self.target_sr  # Update sample rate

            samples_per_segment = int(sample_rate * self.sample_length)
            overlap_samples = int(samples_per_segment * (self.overlap / 100))
            step_size = samples_per_segment - overlap_samples  # Correctly calculates step size for intended overlap

            segments = []
            start = 0
            while start + samples_per_segment <= len(audio_data):
                segment_data = audio_data[start:start + samples_per_segment]
                segments.append(segment_data)
                #print(f"Segment {len(segments)}: Start sample {start}, End sample {start + samples_per_segment}")
                start += step_size  # Move forward based on intended overlap
            if start < len(audio_data):
                segment_data = audio_data[start:]
                padding = samples_per_segment - len(segment_data)
                segment_data = np.pad(segment_data, (0, padding), mode='constant')
                segments.append(segment_data)

        return segments, sample_rate

    def generate_spectrogram(self, audio_segment, sample_rate):
        # Normalize audio to range [-1, 1]
        audio_float = audio_segment.astype(np.float32) / (np.max(np.abs(audio_segment)) + np.finfo(np.float32).eps)

        sgrm = librosa.feature.melspectrogram(y=audio_float, sr=sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.num_mels)
        return librosa.amplitude_to_db(sgrm, ref=np.max)

    def process_wav_for_model(self):
        #print(f"Processing file: {wav_file}")
        if self.file_path is None:
            raise RuntimeError("No file path specified. Please set a file path before processing.")

        segments, sample_rate = self.split_wav()

        spectrograms = []
        for segment in segments:
            sgrm_db = self.generate_spectrogram(segment, sample_rate)
            spectrograms.append(sgrm_db)

        spectrograms = np.array(spectrograms)  # Convert list
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)  # Add channel dim for CNN
    
    def submit_spectrogram_batches(self, aws_access_key_id, aws_secret_access_key):
        """
        Submits spectrogram segments to the endpoint in safe-sized batches.
        Handles JSON responses, preserving order by segment.
        
        Returns a dictionary of segment-level predictions (in order).
        """
        if self.spectrograms is None:
            raise RuntimeError("No spectrograms to submit. Run process_wav_for_model first.")

        # Convert to NumPy if it's a PyTorch tensor
        spectrograms = self.spectrograms
        if hasattr(spectrograms, 'numpy'):
            spectrograms = spectrograms.numpy()

        runtime = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name="us-west-2"
        ).client('sagemaker-runtime')

        responses = []
        total_segments = len(spectrograms)
        #print(f"Total segments: {total_segments}")

        for start in range(0, total_segments, self.max_segments_per_request):
            end = min(start + self.max_segments_per_request, total_segments)
            chunk = spectrograms[start:end]

            # Serialize to .npy format as expected by the endpoint
            buffer = io.BytesIO()
            np.save(buffer, chunk)
            buffer.seek(0)

            try:
                #print(f"Submitting segments {start} to {end} (chunk size: {len(chunk)})")
                response = runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/x-npy',
                    Accept='application/json',
                    Body=buffer.read()
                )

                result = response['Body'].read().decode('utf-8')
                parsed_json = json.loads(result)
                #print(parsed_json)
                responses.append(parsed_json)
            except Exception as e:
                raise RuntimeError(f"Failed to invoke endpoint for batch {start}-{end}: {e}")

        #print('\n\nFull responses list:\n\n'+str(responses))

        # Merge all responses into a single ordered dict
        combined = {}
        global_segment_index = 1

        for batch in responses:
            for local_key in batch:
                combined[f"Segment {global_segment_index}"] = batch[local_key]
                global_segment_index += 1

        print('\n\Modified responses list:\n\n'+str(combined))
        self.raw_predictions = combined
    
    def categorize_effect_confidence(self):
        """
        Attaches a confidence level to each effect based on thresholds passed. Also passes
        through the predicted probabilities. Applies filtering by effect family to
        make sure mutually exclusive effects are not returned.

        Arguments:
        raw_preds - endpoint JSON response.
        thresholds - dictionary containing sets of threshold values for each effect.
        families - dictionary containing lists of mutually exclusive effects.

        Returns:
        Dictionary of segment-level predictions.
        """
        
        if self.raw_predictions is None:
            raise RuntimeError("No raw predictions to categorize. Run submit_spectrogram_batches first.")

        output = {}

        for segment, (top_preds, prob_dict) in self.raw_predictions.items():
            temp_effects = {}
            overruled_info = []
            suppressed_effects = set()  # Keep track of overruled effects
            used_effects = set()        # Track effects that were selected for final output

            # Apply thresholds to get raw effect classifications
            for effect, prob in prob_dict.items():
                if effect not in self.thresholds:
                    continue
                uncertain, confident, very_confident = self.thresholds[effect]

                if prob >= very_confident:
                    level = "very confident"
                elif prob >= confident:
                    level = "confident"
                elif prob >= uncertain:
                    level = "uncertain"
                else:
                    continue

                temp_effects[effect] = {
                    "confidence": level,
                    "value": round(prob, 4)
                }

            final_effects = {}

            # Resolve mutually exclusive families
            for family_name, family_members in self.families.items():
                present = {
                    effect: data for effect, data in temp_effects.items()
                    if effect in family_members
                }

                if len(present) > 1:
                    # Choose best effect in family
                    best_effect, best_data = max(
                        present.items(), key=lambda x: x[1]["value"]
                    )
                    final_effects[best_effect] = best_data
                    used_effects.add(best_effect)

                    excluded = [
                        {"effect": e, "confidence": data["confidence"], "value": data["value"]}
                        for e, data in present.items()
                        if e != best_effect
                    ]

                    suppressed_effects.update([e["effect"] for e in excluded])

                    overruled_info.append({
                        "family": family_name,
                        "chosen": best_effect,
                        "excluded": excluded
                    })

                elif len(present) == 1:
                    only_effect = list(present.items())[0]
                    final_effects[only_effect[0]] = only_effect[1]
                    used_effects.add(only_effect[0])

            # Add non-family effects that aren't overruled
            for effect, data in temp_effects.items():
                if effect not in used_effects and effect not in suppressed_effects:
                    final_effects[effect] = data

            output[segment] = {
                "effects": final_effects,
                "overruled": overruled_info
            }

        self.summarized_segment_results = output

    def summarize_for_llm(self):
        """
        Summarizes confirmed effects across segments and separates them into
        likely_present and possibly_present categories, returning at most 3 effects total.

        Args:
            categorized_results (dict): Output from categorize_effect_confidence().

        Returns:
            Plain-language summary of effects for LLM prompting.
        """
        if self.summarized_segment_results is None:
            raise RuntimeError("No summarized results to analyze. Run categorize_effect_confidence first.")

        segment_data = list(self.summarized_segment_results.values())
        segment_count = len(segment_data)

        effect_counts = defaultdict(Counter)
        segment_effect_map = [set(segment_data[i]["effects"].keys()) for i in range(segment_count)]
        effect_presence_by_segment = defaultdict(list)
        effect_confidence = defaultdict(list)

        for i in range(segment_count):
            for effect in segment_effect_map[i]:
                effect_presence_by_segment[effect].append(i)
                effect_confidence[effect].append(segment_data[i]["effects"][effect]["confidence"])
                confidence = segment_data[i]["effects"][effect]["confidence"]
                effect_counts[effect][confidence] += 1

        likely_present = set()
        for effect, segments in effect_presence_by_segment.items():
            for i in range(1, len(segments)):
                if segments[i] == segments[i - 1] + 1:
                    if "very confident" in effect_confidence[effect]:
                        likely_present.add(effect)
                        break

        all_effects = set(effect_presence_by_segment.keys())
        possibly_present = all_effects - likely_present

        summary_lines = []

        def sort_by_segment_ratio(effects):
            return sorted(
                effects,
                key=lambda eff: len(effect_presence_by_segment[eff]) / segment_count,
                reverse=True
            )

        sorted_likely = sort_by_segment_ratio(likely_present)
        sorted_possible = sort_by_segment_ratio(possibly_present)

        combined_top = sorted_likely[:3] + sorted_possible[:max(0, 3 - len(sorted_likely))]

        final_likely = [eff for eff in sorted_likely if eff in combined_top]
        final_possible = [eff for eff in sorted_possible if eff in combined_top]
        reported_effects = final_likely + final_possible

        if final_likely:
            summary_lines.append("\nLikely present effects:")
            for effect in final_likely:
                levels = effect_counts[effect]
                total = sum(levels.values())
                line = f"- {effect.capitalize()}: appeared in {total}/{segment_count} segments"
                if levels:
                    line += " (" + ", ".join([f"{k}: {v}" for k, v in levels.items()]) + ")"
                summary_lines.append(line)

        if final_possible:
            summary_lines.append("\nPossibly present effects:")
            for effect in final_possible:
                levels = effect_counts[effect]
                total = sum(levels.values())
                line = f"- {effect.capitalize()}: appeared in {total}/{segment_count} segments"
                if levels:
                    line += " (" + ", ".join([f"{k}: {v}" for k, v in levels.items()]) + ")"
                summary_lines.append(line)

        self.predictions_summary_for_llm = "\n".join(summary_lines)
        self.top_effects = reported_effects

        return self.raw_predictions, self.predictions_summary_for_llm, self.summarized_segment_results, self.top_effects
    
    def chatgpt_prompt(self):
        """
        Function prompts ChatGPT to return structure JSON for summarizing effects.
        """
        if self.top_effects is None:
            raise RuntimeError("No top effects to summarize. Run summarize_for_llm first.")

        self.effect_json = self.chat_agent.summarize_effects(self.top_effects, self.predictions_summary_for_llm)
    
    def parse_effects_json(self):
        """
        Converts a JSON structure with a list of effects into a dictionary 
        where each effect name is a key and the value is a dictionary of details.
        """
        if self.effect_json is None:
            raise RuntimeError("No effects JSON to parse. Run chatgpt_prompt first.")
        
        user_education = {}

        for effect in self.effect_json.get("effects", []):
            name = effect.get("name")
            if name:
                # Exclude 'name' from nested details
                details = {k: v for k, v in effect.items() if k != "name"}
                user_education[name.lower()] = details  # Use lowercase keys for consistency

        return user_education