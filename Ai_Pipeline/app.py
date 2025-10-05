# -------------------------------------------------------------------------- #
# UNIFIED AI SERVICE V4.0 (Local Deployment - Color-Enhanced Segmentation)
# -------------------------------------------------------------------------- #
# This service uses DINOv2 for image embeddings and BGE for text embeddings.
# - The segmentation prompt now includes colors for better accuracy.
# - Optimized for local machine deployment with virtual environment.
# - Segmented images are saved to Segmented_Images folder for debugging.
# --------------------------------------------------------------------------
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import numpy as np
import requests
import cv2
import traceback
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
from datetime import datetime, timedelta

# --- Import Deep Learning Libraries ---
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from segment_anything import SamPredictor, sam_model_registry
# --- MODIFIED IMPORTS ---
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

# ==========================================================================
# --- CONFIGURATION & INITIALIZATION ---
# ==========================================================================

app = Flask(__name__)

TEXT_FIELDS_TO_EMBED = ["brand", "material", "markings"]
SCORE_WEIGHTS = {
    "text_score": 0.6,
    "image_score": 0.4
}
FINAL_SCORE_THRESHOLD = 0.75

# Define the Segmented_Images folder path (parallel to Ai_Pipeline)
SEGMENTED_IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Segmented_Images')

# Ensure the Segmented_Images folder exists
if not os.path.exists(SEGMENTED_IMAGES_FOLDER):
    os.makedirs(SEGMENTED_IMAGES_FOLDER, exist_ok=True)
    print(f"📁 Created Segmented_Images folder at: {SEGMENTED_IMAGES_FOLDER}")
else:
    print(f"📁 Using existing Segmented_Images folder at: {SEGMENTED_IMAGES_FOLDER}")

print("="*50)
print("🚀 Initializing Local AI Service with DINOv2...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧠 Using device: {device}")

print("...Loading BGE text model...")
bge_model_id = "BAAI/bge-small-en-v1.5"
tokenizer_text = AutoTokenizer.from_pretrained(bge_model_id)
model_text = AutoModel.from_pretrained(bge_model_id).to(device)
print("✅ BGE model loaded.")

print("...Loading DINOv2 model...")
dinov2_model_id = "facebook/dinov2-base"
processor_dinov2 = AutoImageProcessor.from_pretrained(dinov2_model_id)
model_dinov2 = AutoModel.from_pretrained(dinov2_model_id).to(device)
print("✅ DINOv2 model loaded.")

print("...Loading Grounding DINO model for segmentation...")
gnd_model_id = "IDEA-Research/grounding-dino-base"
# --- MODIFIED PROCESSOR AND MODEL LOADING ---
processor_gnd = GroundingDinoProcessor.from_pretrained(gnd_model_id)
model_gnd = GroundingDinoForObjectDetection.from_pretrained(gnd_model_id).to(device)
print("✅ Grounding DINO model loaded.")

print("...Loading SAM model...")
sam_checkpoint = "sam_vit_b_01ec64.pth"
if not os.path.exists(sam_checkpoint):
    print(f"❌ ERROR: SAM checkpoint file '{sam_checkpoint}' not found!")
    print("   Please ensure sam_vit_b_01ec64.pth is in the Ai_Pipeline directory.")
    sys.exit(1)
sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam_model)
print("✅ SAM model loaded.")
print("="*50)

# ==========================================================================
# --- HELPER FUNCTIONS ---
# ==========================================================================

def get_text_embedding(text: str) -> list:
    if isinstance(text, list):
        if not text: return None
        text = ", ".join(text)
    if not text or not text.strip():
        return None
    instruction = "Represent this sentence for searching relevant passages: "
    inputs = tokenizer_text(instruction + text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model_text(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()[0].tolist()

def get_image_embedding(image: Image.Image) -> list:
    inputs = processor_dinov2(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_dinov2(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy()[0].tolist()

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def jaccard_similarity(set1, set2):
    if not isinstance(set1, set) or not isinstance(set2, set):
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0 if not intersection else 0.0
    return len(intersection) / len(union)

def segment_guided_object(image: Image.Image, object_label: str, colors: list = []) -> Image.Image:
    """
    Finds and segments ALL instances of an object based on a text label and colors,
    returning the original image with the detected objects segmented with transparency.
    This version includes a hole-filling step to create solid masks.
    """
    # Ensure colors is a list
    if isinstance(colors, str):
        colors = [colors] if colors else []
    elif not isinstance(colors, list):
        colors = []
    
    # Create a more descriptive prompt using colors
    color_str = " ".join(c.lower() for c in colors if c)
    if color_str:
        prompt = f"a {color_str} {object_label}."
    else:
        prompt = f"a {object_label}."

    print(f"  [Segment] Using prompt: '{prompt}' for segmentation.")
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    height, width = image_np.shape[:2]

    # Grounding DINO detection
    inputs = processor_gnd(images=image_rgb, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_gnd(**inputs)

    # Process results with a threshold
    results = processor_gnd.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.35, text_threshold=0.5, target_sizes=[(height, width)]
    )

    if not results or len(results[0]['boxes']) == 0:
        print(f"  [Segment] ⚠ Warning: Could not detect '{object_label}' with GroundingDINO. Returning original image.")
        return Image.fromarray(np.concatenate([image_np, np.full((height, width, 1), 255, dtype=np.uint8)], axis=-1), 'RGBA')

    boxes = results[0]['boxes']
    scores = results[0]['scores']
    print(f"  [Segment] ✅ Found {len(boxes)} potential object(s) with confidence scores: {[round(s.item(), 2) for s in scores]}")

    # Set image for SAM
    sam_predictor.set_image(image_np)

    # Initialize an empty mask to combine all detections
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Predict masks for all detected boxes and combine them
    for box in boxes:
        box = box.cpu().numpy().astype(int)
        masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
        combined_mask = np.bitwise_or(combined_mask, masks[0]) # Combine masks
    
    print("  [Segment] Combined masks for all detected objects.")

    # --- START: HOLE FILLING LOGIC ---
    # This new block will fill any holes within the combined mask.
    print("  [Segment] Post-processing: Filling holes in the combined mask...")
    
    # Find contours. RETR_EXTERNAL retrieves only the extreme outer contours.
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new blank mask to draw the filled contours on.
    filled_mask = np.zeros_like(combined_mask)
    
    if contours:
        # Draw the detected contours onto the new mask and fill them.
        # The -1 index means draw all contours, and cv2.FILLED fills them.
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    else:
        # If for some reason no contours were found, fall back to the original mask.
        filled_mask = combined_mask
    print("  [Segment] ✅ Hole filling complete.")
    # --- END: HOLE FILLING LOGIC ---

    # Create an RGBA image where the background is transparent
    object_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    object_rgba[:, :, :3] = image_np # Copy original RGB
    
    # Apply the NEW filled mask as the alpha channel
    object_rgba[:, :, 3] = filled_mask

    return Image.fromarray(object_rgba, 'RGBA')

def save_segmented_image(image: Image.Image, object_name: str) -> str:
    """
    Saves a segmented image to the Segmented_Images folder with a unique filename.
    Returns the saved file path.
    """
    try:
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_object_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in object_name)
        filename = f"{safe_object_name}_{timestamp}.png"
        filepath = os.path.join(SEGMENTED_IMAGES_FOLDER, filename)
        
        # Save the image
        image.save(filepath, format='PNG')
        print(f"    - 💾 Saved segmented image to: {filename}")
        return filepath
    except Exception as e:
        print(f"    - ⚠ Warning: Could not save segmented image: {e}")
        return None

# ==========================================================================
# --- FLASK ENDPOINTS ---
# ==========================================================================

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Unified AI Service is running"}), 200

@app.route('/process', methods=['POST'])
def process_item():
    try:
        data = request.json
        print(f"\n[PROCESS] Received request for: {data.get('objectName')}")

        response = {
            "canonicalLabel": data.get('objectName', '').lower().strip(),
            "brand_embedding": get_text_embedding(data.get('brand')),
            "material_embedding": get_text_embedding(data.get('material')),
            "markings_embedding": get_text_embedding(data.get('markings')),
        }

        image_embeddings = []
        if data.get('images'):
            print(f"  [PROCESS] Processing {len(data['images'])} image(s)...")
            for image_url in data['images']:
                try:
                    img_response = requests.get(image_url, timeout=20)
                    img_response.raise_for_status()
                    image = Image.open(BytesIO(img_response.content))

                    # --- UPDATED: Pass colors to the segmentation function ---
                    segmented_image = segment_guided_object(image, data['objectName'], data.get('colors', []))
                    print(f"    - ✅ Segmentation completed successfully")
                    
                    # Save the segmented image to disk
                    save_segmented_image(segmented_image, data['objectName'])

                    embedding = get_image_embedding(segmented_image)
                    image_embeddings.append(embedding)
                except Exception as e:
                    print(f"    - ⚠ Could not process image {image_url}: {e}")
                    continue

        response["image_embeddings"] = image_embeddings
        print(f"  [PROCESS] ✅ Successfully processed all features.")
        return jsonify(response), 200

    except Exception as e:
        print(f"❌ Error in /process: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_items():
    try:
        payload = request.json
        query_item = payload['queryItem']
        search_list = payload['searchList']
        print(f"\n[COMPARE] Received {len(search_list)} pre-filtered candidates for '{query_item.get('objectName')}'.")

        results = []
        for item in search_list:
            item_id = item.get('_id')
            print(f"\n  - Comparing with item: {item_id}")
            try:
                text_score_components = []
                component_log = {}

                # 1. Calculate score for fields with text embeddings (now includes 'markings')
                for field in TEXT_FIELDS_TO_EMBED:
                    q_emb = query_item.get(f"{field}_embedding")
                    i_emb = item.get(f"{field}_embedding")
                    if q_emb and i_emb: 
                        score = cosine_similarity(q_emb, i_emb)
                        text_score_components.append(score)
                        component_log[field] = f"{score:.4f}"

                # 2. Calculate Jaccard score for 'colors'
                q_colors = set(c.lower().strip() for c in query_item.get('colors', []) if c)
                i_colors = set(c.lower().strip() for c in item.get('colors', []) if c)
                if q_colors and i_colors:
                    score = jaccard_similarity(q_colors, i_colors)
                    text_score_components.append(score)
                    component_log['colors'] = f"{score:.4f}"

                # 3. Calculate direct match score for 'size'
                q_size = (query_item.get('size') or "").lower().strip()
                i_size = (item.get('size') or "").lower().strip()
                if q_size and i_size:
                    score = 1.0 if q_size == i_size else 0.0
                    text_score_components.append(score)
                    component_log['size'] = f"{score:.4f}"

                # 4. Average only the scores from the available components
                text_score = 0.0
                if text_score_components:
                    text_score = sum(text_score_components) / len(text_score_components)
                
                print(f"    - Text Score Components: {component_log}")
                print(f"    - Final Avg Text Score: {text_score:.4f} (from {len(text_score_components)} components)")

                # 5. Calculate Image Score
                image_score = 0.0
                query_img_embs = query_item.get('image_embeddings', [])
                item_img_embs = item.get('image_embeddings', [])
                if query_img_embs and item_img_embs:
                    all_img_scores = []
                    for q_emb in query_img_embs:
                        for i_emb in item_img_embs:
                            all_img_scores.append(cosine_similarity(q_emb, i_emb))
                    if all_img_scores:
                        image_score = max(all_img_scores)
                print(f"    - Max Image Score: {image_score:.4f}")

                # 6. Calculate Final Score (Dynamic)
                final_score = 0.0
                if query_img_embs and item_img_embs:
                    print(f"    - Calculating Hybrid Score (Text + Image)...")
                    final_score = (SCORE_WEIGHTS['text_score'] * text_score + SCORE_WEIGHTS['image_score'] * image_score)
                else:
                    print(f"    - One or both items missing images. Using Text Score only...")
                    final_score = text_score

                print(f"    - Final Dynamic Score: {final_score:.4f}")

                if final_score >= FINAL_SCORE_THRESHOLD:
                    print(f"    - ✅ ACCEPTED (Score >= {FINAL_SCORE_THRESHOLD})")
                    results.append({ "_id": str(item_id), "score": round(final_score, 4) })
                else:
                    print(f"    - ❌ REJECTED (Score < {FINAL_SCORE_THRESHOLD})")

            except Exception as e:
                print(f"    - ⚠ Skipping item {item_id} due to scoring error: {e}")
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n[COMPARE] ✅ Search complete. Found {len(results)} potential matches.")
        return jsonify({"matches": results}), 200

    except Exception as e:
        print(f"❌ Error in /compare: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on local machine - accessible at http://localhost:5000
    print("\n" + "="*50)
    print("🎉 AI Service Ready!")
    print("📍 Server will run at: http://localhost:5000")
    print("📍 Health check: http://localhost:5000/")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)