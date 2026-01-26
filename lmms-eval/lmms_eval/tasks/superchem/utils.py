import re
import io
from PIL import Image

PROMPT_TEMPLATE = r"""Question:

{question}

{options}

Select the best answer to the above multiple-choice question based on the image. Respond with only the letter of the correct option."""

def get_options_string(doc, lang="en"):
    options_dict = doc.get(f'options_{lang}', {})
    options_str = ''
    for key in sorted(options_dict.keys()):
        if options_dict.get(key) is not None:
            options_str += f"{key}: {options_dict[key]}\n"
        else:
            break
    return options_str

def doc_to_visual(doc, kwargs=None):
    lang = "en"
    question = doc.get(f'question_{lang}', '')
    options = get_options_string(doc, lang)
    prompt = PROMPT_TEMPLATE.replace('{question}', question).replace('{options}', options)
    
    question_images = doc.get('question_images') or {}
    options_images = doc.get('options_images') or {}
    images_dict = {**question_images, **options_images}

    visuals = []
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'
    link_pattern = r'!?\s*\[(.+)\]\s*\(([^)]+)\)'
    
    for tag_match in re.finditer(tag_pattern, prompt, re.IGNORECASE | re.DOTALL):
        tag_content = tag_match.group(1)
        link_match = re.match(link_pattern, tag_content, flags=re.MULTILINE | re.DOTALL)
        
        if link_match:
            image_url = link_match.group(2)
            image_bytes = images_dict.get(image_url)
            
            if image_bytes is None:
                raise FileNotFoundError(f"Image file not found: {image_url}")
            
            if image_bytes.startswith(b'\xff\xd8\xff'):
                mimetype = 'jpeg'
            elif image_bytes.startswith(b'\x89PNG'):
                mimetype = 'png'
            else:
                raise ValueError(f"Unsupported image format for {image_url}")

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            visuals.append(img)
        else:
            raise ValueError(f"No link found in multimodal tag: {tag_content}")
            
    return visuals

def doc_to_text(doc, kwargs=None):
    lang = "en"
    question = doc.get(f'question_{lang}', '')
    options = get_options_string(doc, lang)
    prompt = PROMPT_TEMPLATE.replace('{question}', question).replace('{options}', options)
    
    tag_pattern = r'<MultiModal>(.*?)</MultiModal>'

    text = re.sub(tag_pattern, "<image>", prompt, flags=re.IGNORECASE | re.DOTALL)
    return text

def doc_to_target(doc, kwargs=None):
    return {
        "ref": doc.get('answer_en', []),
        "type": doc.get('question_type', '')
    }

def mcp_metric(results, data):
    total_score = 0
    for response, target in zip(results, data):
        pattern = r'\\boxed\{(.+?)\}'
        matches = list(re.finditer(pattern, response, re.DOTALL))
        
        if not matches: continue
        
        answer = matches[-1].group(1).strip()
        for char in ['\\', '{', '}', 'text', 'math', 'bf', 'rm']:
            answer = answer.replace(char, '')
        answer = answer.strip()

        ref_answer = target["ref"]
        q_type = target["type"]
        
        if q_type == 'multiple_choice':
            if len(answer) == len(ref_answer) and all(a.upper() in ref_answer for a in answer):
                total_score += 1
        elif q_type == 'fill_blank':
            if ref_answer and answer.lower() == ref_answer[0].lower():
                total_score += 1
                
    return total_score / len(results) if results else 0

def process_results(doc, results):
    model_output = str(results[0]) if results else ""
    
    pattern = r'\\boxed\{(.+?)\}'
    matches = list(re.finditer(pattern, model_output, re.DOTALL))
    
    content = matches[-1].group(1).strip() if matches else model_output.strip()
    
    for char in ['\\', '{', '}', 'text', 'math', 'bf', 'rm']:
        content = content.replace(char, '')
    for char in ["[", "]", "'", '"', ","]:
        content = content.replace(char, '')
    
    parsed_answer = "".join(re.findall(r'[A-Za-z0-9]', content))

    score = 0
    ref_answer = doc.get('answer_en', [])
    q_type = doc.get('question_type', 'multiple_choice')
    
    if q_type == 'multiple_choice':
        if len(parsed_answer) == len(ref_answer) and all(l.upper() in ref_answer for l in parsed_answer):
            score = 1
    elif q_type == 'fill_blank':
        if ref_answer and parsed_answer.lower() == str(ref_answer[0]).lower():
            score = 1

    return {
        "mcp_accuracy": score,
        "groundtruth": ref_answer,
        "raw_output": model_output,
        "question": doc_to_text(doc),
        "parsed_answer": parsed_answer
    }