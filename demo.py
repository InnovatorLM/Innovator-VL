#!/usr/bin/env python3
"""
Innovator-VL Inference Demo
A simple inference script to demonstrate Innovator-VL multimodal capabilities.
"""

import argparse
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

def parse_arguments():
    parser = argparse.ArgumentParser(description="Innovator-VL Inference Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Innovator-VL-8B-Instruct",
        help="Path to Innovator-VL model"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="/path/to/image.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Describe this image.",
        help="Query text"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": args.query},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("\n" + "="*50)
    print("Response:")
    print("="*50)
    print(output_text[0])
    print("="*50)

if __name__ == "__main__":
    main()
