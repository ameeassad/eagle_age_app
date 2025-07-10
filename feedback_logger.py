import json
from datetime import datetime
import streamlit as st

def log_feedback(prediction, feedback_type, image_filename=None):
    """
    Log user feedback to a JSON file for developer analysis.
    
    Args:
        prediction (str): The model's prediction (e.g., '1K', '2K', etc.)
        feedback_type (str): User feedback ('correct', 'incorrect', 'unsure')
        image_filename (str, optional): Name of the uploaded image file
    """
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "feedback": feedback_type,
        "image_filename": image_filename
    }
    
    try:
        # Load existing feedback or create new file
        try:
            with open('feedback_log.json', 'r') as f:
                feedback_log = json.load(f)
        except FileNotFoundError:
            feedback_log = []
        
        # Add new feedback
        feedback_log.append(feedback_data)
        
        # Save updated log
        with open('feedback_log.json', 'w') as f:
            json.dump(feedback_log, f, indent=2)
            
    except Exception as e:
        st.error(f"Could not save feedback: {e}")

def get_feedback_stats():
    """
    Get statistics from the feedback log.
    
    Returns:
        dict: Statistics about feedback including counts and accuracy
    """
    try:
        with open('feedback_log.json', 'r') as f:
            feedback_log = json.load(f)
        
        if not feedback_log:
            return {"total_feedback": 0}
        
        total = len(feedback_log)
        correct = sum(1 for entry in feedback_log if entry["feedback"] == "correct")
        incorrect = sum(1 for entry in feedback_log if entry["feedback"] == "incorrect")
        unsure = sum(1 for entry in feedback_log if entry["feedback"] == "unsure")
        
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        
        return {
            "total_feedback": total,
            "correct": correct,
            "incorrect": incorrect,
            "unsure": unsure,
            "accuracy": accuracy
        }
        
    except FileNotFoundError:
        return {"total_feedback": 0}
    except Exception as e:
        st.error(f"Could not read feedback log: {e}")
        return {"total_feedback": 0}