import subprocess
import pandas as pd
import os
import sys

# --- Test Configuration ---
MENTEE_TEST_FILE = './data/mentee_test.xlsx'
MENTOR_TEST_FILE = './data/mentor_test.xlsx'
OUTPUT_TEST_FILE = 'test_matches.xlsx'
MENTMATCH_SCRIPT_PATH = './code/mentmatch.py' # Relative path to the main script

# Define "ideal" matches based on manual inspection of test data
# Format: {mentee_id: expected_mentor_id}
# Based on mentee_test.xlsx and mentor_test.xlsx:
# - Alice (101, Python/Data) matches Frank (201, Python/Data)
# - Bob (102, Proj Mgmt) matches Grace (202, Proj Mgmt)
# - Charlie (103, Public Speaking) matches Heidi (203, Public Speaking)
# - David (104, Cloud/DevOps) matches Ivan (204, Cloud/DevOps)
# - Eve (105, ML/AI) - No direct mentor match in competencies, Frank (201) might be closest due to capacity and data science overlap, or unassigned.
# Let's assume for this test, we expect the specific competency matches:
IDEAL_MATCHES = {
    101: 201, # Alice -> Frank
    102: 202, # Bob -> Grace
    103: 203, # Charlie -> Heidi
    104: 204, # David -> Ivan
    # Eve (105) is harder to define an "ideal" match for strictly based on primary competency
}

def run_mentmatch():
    """Runs the main mentmatch script as a subprocess within the 'gt' conda environment."""
    # Use 'conda run' to execute within the specified environment
    command = [
        'conda', 'run', '-n', 'gt', # Specify conda environment
        'python', # The command to run within the environment
        MENTMATCH_SCRIPT_PATH,
        '--mentee_file', MENTEE_TEST_FILE,
        '--mentor_file', MENTOR_TEST_FILE,
        '--output_file', OUTPUT_TEST_FILE
    ]
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("MentMatch script ran successfully.")
        print("--- Script Output ---")
        print(result.stdout)
        print("---------------------")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running MentMatch script: {e}")
        print("--- Script Stderr ---")
        print(e.stderr)
        print("---------------------")
        print("--- Script Stdout ---")
        print(e.stdout)
        print("---------------------")
        return False
    except FileNotFoundError:
        print(f"Error: Could not find the script '{MENTMATCH_SCRIPT_PATH}' or the 'conda' command.")
        print("Ensure conda is installed and in your PATH, and the 'gt' environment exists.")
        return False

def evaluate_results():
    """Loads the results and calculates performance metrics."""
    if not os.path.exists(OUTPUT_TEST_FILE):
        print(f"Error: Output file '{OUTPUT_TEST_FILE}' not found. Cannot evaluate.")
        return

    try:
        df_results = pd.read_excel(OUTPUT_TEST_FILE)
    except Exception as e:
        print(f"Error reading results file '{OUTPUT_TEST_FILE}': {e}")
        return

    # --- Basic Metrics Calculation ---
    num_mentees = len(IDEAL_MATCHES) # Or load from mentee file if needed
    correct_matches = 0
    assigned_mentees = 0
    total_similarity = 0.0
    mentee_assignments = {}

    # Check if expected columns exist
    required_cols = ['mentee_id', 'match_1_mentor_id', 'match_1_score'] # Assuming top match is the assignment
    if not all(col in df_results.columns for col in required_cols):
        print(f"Error: Results file '{OUTPUT_TEST_FILE}' is missing required columns. Found: {df_results.columns.tolist()}")
        return

    for _, row in df_results.iterrows():
        mentee_id = row['mentee_id']
        assigned_mentor_id = row['match_1_mentor_id'] # Assuming first match is the final assignment
        similarity_score = row['match_1_score']

        if pd.notna(assigned_mentor_id):
            assigned_mentees += 1
            total_similarity += similarity_score
            mentee_assignments[mentee_id] = int(assigned_mentor_id)

            # Check against ideal matches
            if mentee_id in IDEAL_MATCHES and IDEAL_MATCHES[mentee_id] == int(assigned_mentor_id):
                correct_matches += 1
        else:
             mentee_assignments[mentee_id] = None # Mark as unassigned

    # --- Performance Metrics ---
    # Accuracy: Correct assignments / number of mentees with an ideal match defined
    accuracy = (correct_matches / len(IDEAL_MATCHES)) * 100 if IDEAL_MATCHES else 0
    # Coverage: Mentees assigned / total number of test mentees (could use len(df_results) or len(mentee_data) for total)
    # Let's use the number of rows in the results file as the total number of mentees processed.
    total_processed_mentees = len(df_results)
    coverage = (assigned_mentees / total_processed_mentees) * 100 if total_processed_mentees > 0 else 0
    # Average Similarity: Total similarity / number of assigned mentees
    avg_similarity = total_similarity / assigned_mentees if assigned_mentees > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Mentee Assignments (Generated): {mentee_assignments}")
    print(f"Ideal Assignments (Expected):   {IDEAL_MATCHES}")
    print("--------------------------")
    print(f"Total Mentees Processed: {total_processed_mentees}")
    print(f"Mentees Assigned a Mentor: {assigned_mentees}")
    print(f"Mentees with Defined Ideal Match: {len(IDEAL_MATCHES)}")
    print(f"Correct Matches (vs Ideal): {correct_matches}")
    print("--- Performance Metrics ---")
    print(f"Accuracy (vs Ideal): {accuracy:.2f}%")
    print(f"Coverage (Assigned/Total): {coverage:.2f}%")
    print(f"Average Similarity Score (for assigned): {avg_similarity:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    if run_mentmatch():
        evaluate_results()
    else:
        print("Skipping evaluation due to script execution failure.")
