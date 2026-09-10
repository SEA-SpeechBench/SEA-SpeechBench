"""
F1_Coverage_Purity metric for TLoc (Time Location) tasks.
Evaluates the accuracy of time span predictions in audio content.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def extract_time_span(text: str) -> Optional[Tuple[float, float]]:
    """
    Extract start and end time from text by finding the first two numbers.
    Simple and reliable approach: first number = start, second number = end.
    Special handling for minutes:seconds format (e.g., "0:26").
    
    Args:
        text (str): Text containing time information
        
    Returns:
        Tuple[float, float]: (start_time, end_time) or None if not found
    """
    if not text:
        return None
    
    # Special case: minutes:seconds format (e.g., "0:26" means 0 to 26 seconds)
    time_pattern = r'(\d+):(\d+\.?\d*)'
    time_match = re.search(time_pattern, text)
    if time_match:
        try:
            start = float(time_match.group(1))
            end = float(time_match.group(2))
            # Basic validation: end should be greater than start
            if end > start:
                return (start, end)
        except ValueError:
            pass
    
    # General case: find all numbers in the text (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    
    # Need at least 2 numbers
    if len(numbers) < 2:
        return None
    
    try:
        # First number is start time, second number is end time
        start = float(numbers[0])
        end = float(numbers[1])
        
        # Basic validation: end should be greater than start
        if end <= start:
            return None
            
        # Optional: filter out unreasonable values (e.g., > 10000 seconds)
        if start > 10000 or end > 10000:
            return None
            
        return (start, end)
        
    except ValueError:
        return None


def calculate_overlap(pred_start: float, pred_end: float, ref_start: float, ref_end: float) -> float:
    """
    Calculate the overlap between predicted and reference time spans.
    
    Args:
        pred_start, pred_end: Predicted time span
        ref_start, ref_end: Reference time span
        
    Returns:
        float: Overlap duration
    """
    overlap_start = max(pred_start, ref_start)
    overlap_end = min(pred_end, ref_end)
    return max(0, overlap_end - overlap_start)


def calculate_coverage(pred_start: float, pred_end: float, ref_start: float, ref_end: float) -> float:
    """
    Calculate coverage: how much of the reference span is covered by prediction.
    
    Args:
        pred_start, pred_end: Predicted time span
        ref_start, ref_end: Reference time span
        
    Returns:
        float: Coverage ratio (0-1)
    """
    ref_duration = ref_end - ref_start
    if ref_duration <= 0:
        return 0.0
    
    overlap = calculate_overlap(pred_start, pred_end, ref_start, ref_end)
    return overlap / ref_duration


def calculate_purity(pred_start: float, pred_end: float, ref_start: float, ref_end: float) -> float:
    """
    Calculate purity: how much of the prediction is within the reference span.
    
    Args:
        pred_start, pred_end: Predicted time span
        ref_start, ref_end: Reference time span
        
    Returns:
        float: Purity ratio (0-1)
    """
    pred_duration = pred_end - pred_start
    if pred_duration <= 0:
        return 0.0
    
    overlap = calculate_overlap(pred_start, pred_end, ref_start, ref_end)
    return overlap / pred_duration


def calculate_f1_score(coverage: float, purity: float) -> float:
    """
    Calculate F1 score from coverage and purity.
    
    Args:
        coverage: Coverage score (0-1)
        purity: Purity score (0-1)
        
    Returns:
        float: F1 score (0-1)
    """
    if coverage + purity == 0:
        return 0.0
    return 2 * (coverage * purity) / (coverage + purity)


def f1_coverage_purity(predictions: List[str], references: List[str], 
                      questions: Optional[List[str]] = None,
                      tolerance: float = 1.0) -> Dict[str, Any]:
    """
    Calculate F1_Coverage_Purity metric for TLoc tasks.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        questions: Optional list of questions (not used)
        tolerance: Time tolerance in seconds for exact matches
        
    Returns:
        Dict containing metric scores and details
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    if not predictions:
        return {
            "f1_score": 0.0,
            "coverage": 0.0,
            "purity": 0.0,
            "exact_match": 0.0,
            "total_samples": 0,
            "valid_samples": 0,
            "sample_details": []
        }
    
    total_f1 = 0.0
    total_coverage = 0.0
    total_purity = 0.0
    exact_matches = 0
    valid_samples = 0
    sample_details = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Extract time spans
        pred_span = extract_time_span(pred)
        ref_span = extract_time_span(ref)
        
        sample_detail = {
            "sample_id": i,
            "prediction": pred,
            "reference": ref,
            "pred_span": pred_span,
            "ref_span": ref_span,
            "valid": False,
            "f1": 0.0,
            "coverage": 0.0,
            "purity": 0.0,
            "exact_match": False
        }
        
        if pred_span is None or ref_span is None:
            sample_details.append(sample_detail)
            continue
        
        pred_start, pred_end = pred_span
        ref_start, ref_end = ref_span
        
        # Check for exact match within tolerance
        exact_match = (abs(pred_start - ref_start) <= tolerance and 
                      abs(pred_end - ref_end) <= tolerance)
        
        if exact_match:
            exact_matches += 1
            sample_detail["exact_match"] = True
            sample_detail["f1"] = 1.0
            sample_detail["coverage"] = 1.0
            sample_detail["purity"] = 1.0
        else:
            # Calculate coverage and purity
            coverage = calculate_coverage(pred_start, pred_end, ref_start, ref_end)
            purity = calculate_purity(pred_start, pred_end, ref_start, ref_end)
            f1 = calculate_f1_score(coverage, purity)
            
            sample_detail["f1"] = f1
            sample_detail["coverage"] = coverage
            sample_detail["purity"] = purity
        
        sample_detail["valid"] = True
        valid_samples += 1
        total_f1 += sample_detail["f1"]
        total_coverage += sample_detail["coverage"]
        total_purity += sample_detail["purity"]
        sample_details.append(sample_detail)
    
    # Calculate averages
    if valid_samples > 0:
        avg_f1 = total_f1 / valid_samples
        avg_coverage = total_coverage / valid_samples
        avg_purity = total_purity / valid_samples
        exact_match_rate = exact_matches / valid_samples
    else:
        avg_f1 = avg_coverage = avg_purity = exact_match_rate = 0.0
    
    return {
        "f1_score": avg_f1,
        "coverage": avg_coverage,
        "purity": avg_purity,
        "exact_match": exact_match_rate,
        "total_samples": len(predictions),
        "valid_samples": valid_samples,
        "invalid_samples": len(predictions) - valid_samples,
        "sample_details": sample_details
    }


def f1_coverage_purity_evaluate(self, input_data) -> tuple:
    """
    Evaluation function for the metric registry.
    Returns format compatible with existing score files: (metric_scores, details)
    
    Args:
        self: The metric instance
        input_data: List of [questions, references, predictions] or list of dicts
        
    Returns:
        Tuple of (metric_scores, details) compatible with existing score file format
    """
    # Handle different input formats
    if isinstance(input_data, list) and len(input_data) == 3 and isinstance(input_data[0], list):
        # Format: [questions, references, predictions]
        questions, references, predictions = input_data
    else:
        # Format: list of dicts
        predictions = []
        references = []
        questions = []
        
        for item in input_data:
            predictions.append(item.get("model_prediction", item.get("prediction", "")))
            references.append(item.get("answer", item.get("reference", "")))
            questions.append(item.get("text", item.get("question", "")))
    
    # Get detailed results
    results = f1_coverage_purity(predictions, references, questions)
    
    # Format metric scores to match existing format
    metric_scores = {
        "f1_score": results["f1_score"],
        "coverage": results["coverage"], 
        "purity": results["purity"],
        "exact_match": results["exact_match"]
    }
    
    # Format details to match existing format
    details = []
    for i, detail in enumerate(results["sample_details"]):
        detail_dict = {
            "question": questions[i] if i < len(questions) else "",
            "reference": detail["reference"],
            "model_prediction": detail["prediction"],
            "f1_score": detail["f1"],
            "coverage": detail["coverage"],
            "purity": detail["purity"],
            "exact_match": detail["exact_match"],
            "valid": detail["valid"]
        }
        details.append(detail_dict)
    
    return metric_scores, details
