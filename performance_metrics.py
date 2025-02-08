from pathlib import Path
import psutil
import torch
import threading
import time
import openvino_genai as ov_genai
from transformers import AutoTokenizer  # For token counting
from datasets import load_dataset       # For loading a dataset from Hugging Face

#########################################
# Performance Monitor & Streaming Class #
#########################################

class PerformanceMonitor:
    def __init__(self):
        self.stop_monitor = False
        self.ram_samples = []
        self.cpu_samples = []
        self.gpu_usage = []
        self.vram_usage = []
        self.process = psutil.Process()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def start_monitoring(self):
        """Start monitoring resources in a separate thread"""
        self.monitor_thread = threading.Thread(target=self._collect_metrics)
        self.monitor_thread.start()

    def _collect_metrics(self):
        """Collect metrics until stopped"""
        while not self.stop_monitor:
            self.ram_samples.append(self.process.memory_info().rss / 1024 / 1024)
            self.cpu_samples.append(psutil.cpu_percent())
            if self.device.type == "cuda":
                self.gpu_usage.append(torch.cuda.utilization())
                self.vram_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
            time.sleep(0.1)

    def stop_monitoring(self):
        self.stop_monitor = True
        self.monitor_thread.join()
        return {
            "avg_ram": sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0,
            "peak_ram": max(self.ram_samples) if self.ram_samples else 0,
            "avg_cpu": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            "cores_used": (sum(self.cpu_samples) / len(self.cpu_samples) / 100) * psutil.cpu_count() if self.cpu_samples else 0,
            "avg_gpu": sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0,
            "avg_vram": sum(self.vram_usage) / len(self.vram_usage) if self.vram_usage else 0
        }

class Streamer:
    def __init__(self):
        self.generated_text = ""
        self.first_token_time = None
        self.start_time = None
        self.token_count = 0
        
    def __call__(self, text: str):
        # Print text immediately
        print(text, end='', flush=True)
        
        # Track timing metrics
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        if text.strip() and self.first_token_time is None:
            self.first_token_time = current_time
            
        # Accumulate text for token counting
        self.generated_text += text

#####################################
# Inference Helper Functions        #
#####################################

def run_inference_for_sample(instruction: str, pipe, config, tokenizer):
    """
    Run inference on a single instruction and calculate metrics.
    Returns a tuple of (metrics_dict, generated_text).
    """
    # Initialize a new monitor and streamer for this sample
    monitor = PerformanceMonitor()
    streamer = Streamer()
    
    # Start monitoring and inference timing
    monitor.start_monitoring()
    gen_start = time.time()
    
    # Run inference (streaming the output)
    pipe.generate(instruction, streamer=streamer, max_new_tokens=config.max_new_tokens)
    
    gen_end = time.time()
    sample_metrics = monitor.stop_monitoring()
    
    # Calculate token-based metrics
    generated_text = streamer.generated_text
    tokenized_output = tokenizer.encode(generated_text)
    total_tokens = len(tokenized_output)
    inference_time = gen_end - gen_start
    ttft = (streamer.first_token_time - gen_start) if streamer.first_token_time else 0
    tpot = (inference_time - ttft) / (total_tokens - 1) if total_tokens > 1 else 0
    throughput = total_tokens / inference_time if inference_time > 0 else 0

    # Add these additional metrics to our dictionary
    sample_metrics.update({
        "Total Tokens": total_tokens,
        "Inference Time (s)": inference_time,
        "Time to First Token (s)": ttft,
        "Avg Time Per Token (s)": tpot,
        "Throughput (tokens/s)": throughput
    })
    
    return sample_metrics, generated_text

def evaluate_dataset(pipe, config, tokenizer, dataset, sample_fraction=0.02):
    """
    Evaluate the model on a fraction of the dataset's validation split.
    Prints the prompt, generated text, and metrics for each sample,
    and then computes and prints the average metrics.
    """
    total_samples = len(dataset)
    num_samples_to_evaluate = max(1, int(total_samples * sample_fraction))
    print("=" * 100)
    print(f"Evaluating on {num_samples_to_evaluate} samples out of {total_samples}")
    
    all_metrics = []
    for i in range(num_samples_to_evaluate):
        sample = dataset[i]
        # Try to extract the prompt/instruction from the sample.
        # Adjust the field name if needed.
        prompt = sample.get("instruction", sample.get("prompt", None))
        if prompt is None:
            print(f"Sample {i} does not contain an 'instruction' or 'prompt' field. Skipping.")
            continue

        print("\n" + "=" * 100)
        print(f"Sample {i+1} Prompt:\n{prompt}\n")
        
        # Run inference on the sample
        sample_metrics, generated_text = run_inference_for_sample(prompt, pipe, config, tokenizer)
        
        print("\nGenerated Text:\n")
        print(generated_text)
        print("\nMetrics for Sample {}:".format(i+1))
        for k, v in sample_metrics.items():
            # Format float values to 2 decimal places
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
                
        all_metrics.append(sample_metrics)
    
    # Compute and print average metrics over all evaluated samples
    if all_metrics:
        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            avg_metrics[key] = sum(sample[key] for sample in all_metrics) / len(all_metrics)
        print("\n" + "=" * 100)
        print("Average Metrics over Evaluated Samples:")
        for k, v in avg_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print("=" * 100)
        return avg_metrics
    else:
        print("No valid samples were processed.")
        return None

#######################
# Main Execution Flow #
#######################

def main():
    # -------------------------------
    # 1. Model and Tokenizer Loading
    # -------------------------------
    llm_model_path = './phi-3.5-mini-instruct'
    # llm_model_path = './gemma-2b-9b-it'
    # llm_model_path='./mistral-7b-instruct-v0.3'
    # llm_model_path='./qwen-2.5-14b-instruct-1m'
    # llm_model_path='/home/hetarth2/cropwizard-on-edge/yi-1.5-9b'
    device = 'CPU'
    
    # Measure model load time
    load_start = time.time()
    print("Loading model...")
    pipe = ov_genai.LLMPipeline(llm_model_path, device)
    load_end = time.time()
    load_time = load_end - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Initialize tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    
    # Get and configure generation parameters
    config = pipe.get_generation_config()
    config.max_new_tokens = 128
    config.num_beam_groups = 3
    config.num_beams = 15
    config.diversity_penalty = 1.0

    # -------------------------------
    # 2. Single Sample Inference Demo
    # -------------------------------
    print("\n" + "=" * 100)
    print("Starting single-sample inference...\n")
    
    # Initialize monitor and streamer for the demo sample
    monitor = PerformanceMonitor()
    streamer = Streamer()
    monitor.start_monitoring()
    gen_start = time.time()
    
    # Run inference on a demo prompt
    demo_prompt = "Please write a 200 word essay on UIUC"
    pipe.generate(demo_prompt, streamer=streamer, max_new_tokens=config.max_new_tokens)
    
    gen_end = time.time()
    metrics = monitor.stop_monitoring()
    
    # Calculate token-based metrics for the demo sample
    tokenized_output = tokenizer.encode(streamer.generated_text)
    total_tokens = len(tokenized_output)
    inference_time = gen_end - gen_start
    ttft = streamer.first_token_time - gen_start if streamer.first_token_time else 0
    tpot = (inference_time - ttft) / (total_tokens - 1) if total_tokens > 1 else 0
    throughput = total_tokens / inference_time if inference_time > 0 else 0
    
    print("\n" + "=" * 100)
    print(f"Model Load Time: {load_time:.2f}s")
    print(f"Time to First Token (TTFT): {ttft * 1000:.2f}ms")
    print(f"Avg Time Per Output Token (TPOT): {tpot * 1000:.2f}ms")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Inference Time: {inference_time:.2f}s")
    print(f"Average RAM Usage: {metrics['avg_ram']:.2f} MB")
    print(f"Peak RAM Usage: {metrics['peak_ram']:.2f} MB")
    print(f"Average CPU Usage: {metrics['avg_cpu']:.2f}%")
    print(f"Average Cores Used: {metrics['cores_used']:.2f}")
    if monitor.device.type == "cuda":
        print(f"Average GPU Usage: {metrics['avg_gpu']:.2f}%")
        print(f"Average VRAM Usage: {metrics['avg_vram']:.2f} MB")
    print("=" * 100)
    
    # -------------------------------
    # 3. Evaluate on a Hugging Face Dataset
    # -------------------------------
    dataset_name = "tatsu-lab/alpaca"  
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        dataset = None
    
    if dataset:
        evaluate_dataset(pipe, config, tokenizer, dataset, sample_fraction=0.0001)

if __name__ == "__main__":
    main()


