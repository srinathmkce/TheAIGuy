import os
from collections import defaultdict
import base64
import json
import mlflow
import pandas as pd
from tqdm import tqdm
import io
from PIL import Image
from io import BytesIO


def pil_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_urls(dataset):
    url_list = []
    ground_truth_list = []
    for data in tqdm(dataset):
        image = data['image']
        ground_truth = json.loads(data['ground_truth'])['gt_parse']
        image_base64 = pil_to_base64(image)
        url_list.append(image_base64)
        ground_truth_list.append(ground_truth)
    return url_list, ground_truth_list


def flatten_json(y, prefix=''):
    """Flatten nested JSON into dot notation keys."""
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}.")
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, f"{name}{i}.")
        else:
            out[name[:-1]] = x
    flatten(y, prefix)
    return out


def apply_postprocessing(data):
    # if the key contains price replace comma by . in the price columns
    for key in data.keys():
        if "price" in key.lower() and isinstance(data[key], str):
            data[key] = data[key].replace(",", ".")
    return data


def fix_data_type_mismatch(gt, pred):
    if 'menu' in gt and 'menu' in pred:
        if isinstance(gt['menu'], dict) and isinstance(pred['menu'], list):
            gt['menu'] = [gt['menu']]
        
        if isinstance(gt['menu'], list) and isinstance(pred['menu'], dict):
            pred['menu'] = [pred['menu']]

    return gt, pred

def calculate_invoice_accuracies(ground_truth_list, response_list):
    """Calculate per-invoice accuracy and return a DataFrame."""
    invoice_metrics = []
    for i in range(len(ground_truth_list)):
        gt = ground_truth_list[i]
        pred = response_list[i]
        # If response is an OpenAI object, parse output_text
        if hasattr(pred, "output_text"):
            pred = json.loads(pred.output_text)
        
        gt, pred = fix_data_type_mismatch(gt, pred)
        gt_flat = flatten_json(gt)
        gt_flat = apply_postprocessing(gt_flat)
        pred_flat = flatten_json(pred)
        pred_flat = apply_postprocessing(pred_flat)
        total_keys = len(gt_flat)
        matched_keys = sum(
            str(gt_flat[k]).strip() == str(pred_flat.get(k, "")).strip()
            for k in gt_flat
        )
        accuracy = matched_keys / total_keys if total_keys > 0 else 0.0
        invoice_metrics.append({
            "invoice_no": i,
            "total_keys": total_keys,
            "matched_keys": matched_keys,
            "accuracy": accuracy
        })
    invoice_metrics_df = pd.DataFrame(invoice_metrics)
    invoice_metrics_df.to_csv(os.path.join("artifacts", "invoice_metrics.csv"))
    return invoice_metrics_df


def calculate_key_level_metrics(gt_list, pred_list):
    key_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for gt_json, pred_json in zip(gt_list, pred_list):
        gt_json, pred_json = fix_data_type_mismatch(gt_json, pred_json)
        gt_flat = flatten_json(gt_json)
        gt_flat = apply_postprocessing(gt_flat)
        pred_flat = flatten_json(pred_json)
        pred_flat = apply_postprocessing(pred_flat)
        gt_keys = set(gt_flat.keys())
        pred_keys = set(pred_flat.keys())
        all_keys = gt_keys | pred_keys
        for k in all_keys:
            gt_val = gt_flat.get(k)
            pred_val = pred_flat.get(k)
            if gt_val is not None and pred_val is not None:
                if gt_val == pred_val:
                    key_stats[k]['tp'] += 1
                else:
                    # print(f"False positive for key '{k}': GT='{gt_val}', Pred='{pred_val}'")
                    key_stats[k]['fp'] += 1  # Value present but incorrect
            elif gt_val is not None and pred_val is None:
                # print(f"False negative for key '{k}': GT='{gt_val}', Pred=None")
                key_stats[k]['fn'] += 1   # Value missing in prediction
            elif gt_val is None and pred_val is not None:
                # print(f"False positive for key '{k}': GT=None, Pred='{pred_val}'")
                key_stats[k]['fp'] += 1   # Extra key in prediction
    
    # Calculate metrics per key
    metrics = {}
    for k, stats in key_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[k] = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    key_metrics_df = pd.DataFrame(metrics).T.reset_index(names='key')
    key_metrics_df.to_csv(os.path.join("artifacts", "key_metrics.csv"))
    return key_metrics_df

def calculate_individual_invoice_accuracies(ground_truth, output):
    """Calculate per-invoice accuracy and return a DataFrame."""
    invoice_metrics = []
    ground_truth, output = fix_data_type_mismatch(ground_truth, output)
    gt_flat = flatten_json(ground_truth)
    gt_flat = apply_postprocessing(gt_flat)
    pred_flat = flatten_json(output)
    pred_flat = apply_postprocessing(pred_flat)
    total_keys = len(gt_flat)
    # Create a dataframe with following columns, ground_truth_value, predicted_value, match (True or False) and calculate the overall accuracy based on the match
    for k in gt_flat:
        invoice_metrics.append({
            "key": k,
            "ground_truth_value": gt_flat[k],
            "predicted_value": pred_flat.get(k, ""),
            "match": str(gt_flat[k]).strip() == str(pred_flat.get(k, "")).strip()
        })
    invoice_metrics_df = pd.DataFrame(invoice_metrics)

    accuracy = invoice_metrics_df["match"].mean()
    return invoice_metrics_df, round(accuracy * 100, 2)

def convert_base64_to_pil(image_base64):
    """Convert a base64-encoded image to a PIL Image."""
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    return image

def retrieve_token_usage(trace_df):
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    for i in range(len(trace_df)):
        # For Gemini responses, token usage might be in different format
        try:
            token_dict = trace_df["response"][i]['usage']
            input_tokens = token_dict.get("input_tokens", 0)
            output_tokens = token_dict.get("output_tokens", 0)
            # Gemini doesn't have reasoning tokens like GPT-5
            reasoning_tokens = 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_reasoning_tokens += reasoning_tokens
        except (KeyError, TypeError):
            # If token usage format is different, try alternative parsing
            try:
                # For LangChain responses, token usage might be in usage_metadata
                usage_metadata = trace_df["response"][i].get('usage_metadata', {})
                input_tokens = usage_metadata.get("input_tokens", 0)
                output_tokens = usage_metadata.get("output_tokens", 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except (KeyError, TypeError):
                # If no token usage available, continue with 0
                continue
    return total_input_tokens, total_output_tokens, total_reasoning_tokens