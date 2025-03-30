import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # For cleaning text

# --- Configuration ---
MENTEE_FILE_PATH = 'path/to/your/mentee_data.xlsx' # <--- !!! UPDATE THIS PATH !!!
MENTOR_FILE_PATH = 'path/to/your/mentor_data.xlsx' # <--- !!! UPDATE THIS PATH !!!
OUTPUT_FILE_PATH = 'mentor_mentee_matches.xlsx' # Path to save the results

# Define the TARGET standardized column names we expect to use after renaming
# These correspond to the values in the .rename() dictionary keys used for semantic profiles
MENTEE_TARGET_SEMANTIC_COLS = [
    'competencies_desired',
    'reason_for_participating',
    'hobby',
    'movie_genre',
    'book_genre',
    'fun_fact'
]
MENTOR_TARGET_SEMANTIC_COLS = [
    'competencies_offered',
    'reason_for_participating',
    'hobby',
    'movie_genre',
    'book_genre',
    'fun_fact'
]

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def combine_text_fields(row, columns):
    """Combine text from specified columns into a single string."""
    combined = []
    for col in columns:
        # Check if the target column exists in the row's index (DataFrame columns)
        if col in row.index and pd.notna(row[col]):
            combined.append(str(row[col]))
        # Optional: Add a warning if a column is expected but missing for a row
        # else:
        #     print(f"Warning: Column '{col}' not found or is NaN for row index {row.name}")
    return ". ".join(combined)


def standardize_col_names(df):
    """Standardize column names for easier access."""
    # Step 1: Basic standardization (lowercase, underscore spaces, remove chars)
    original_columns = df.columns.tolist()
    # Remove common punctuation and symbols more broadly
    standardized_columns = [re.sub(r'[?"\':()\[\];,./]', '', col.lower().strip()) for col in original_columns]
    # Replace sequences of whitespace with a single underscore
    standardized_columns = [re.sub(r'\s+', '_', col) for col in standardized_columns]
    # Ensure no double underscores resulted
    standardized_columns = [re.sub(r'_+', '_', col) for col in standardized_columns]
    df.columns = standardized_columns

    # --- DEBUGGING PRINT ---
    print("\n--- Columns after initial standardization (before renaming): ---")
    print(df.columns.tolist())
    # --- END DEBUGGING PRINT ---

    # Step 2: Manual renaming for known variations to TARGET names
    # Keys should be the expected result AFTER the initial standardization above
    # Values are the final names used in the script
    rename_map = {
        # Mentee specific
        'competencies_you_would_like_to_work_on_with_a_mentor': 'competencies_desired',
        'would_you_prefer_to_be_matched_with_a_mentor_in_the_same_amentum_connection_networks_as_you': 'prefer_same_network_mentee',
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mix_of_both': 'personality', # Mentee version
        # Mentor specific
        'what_is_your_current_job_title': 'title',
        'select_your_occupation_category': 'category',
        'what_is_your_current_level_within_the_company': 'level',
        'who_is_your_direct_manager': 'manager', # Mentor's manager
        'in_which_time_zone_do_you_work_currently': 'time_zone',
        'how_many_years_have_you_been_with_amentum': 'years_with_amentum',
        'please_select_the_competencies_you_feel_you_could_teach': 'competencies_offered',
        'how_many_mentees_are_you_able_to_mentor_throughout_the_program': 'mentor_capacity',
        'what_amentum_connect_networks_are_you_a_member_of': 'amentum_connection_networks', # Adjusted key slightly
        'would_you_prefer_to_be_matched_with_a_mentee_in_the_same_amentum_connection_networks_as_you': 'prefer_same_network_mentor',
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mixter_of_both': 'personality', # Mentor version ('mixter')
         # Common columns mapped to target names (keys adjusted based on standardization)
        'name': 'name', # Ensure 'name' isn't accidentally renamed if present in map
        'id': 'id',     # Ensure 'id' isn't accidentally renamed
        'email': 'email', # Keep email if needed
        'manager': 'manager', # Mentee's manager
        'title': 'title', # Mentee's title
        'level': 'level', # Mentee's level
        'category': 'category', # Mentee's category
        'time_zone': 'time_zone', # Mentee's time zone
        'years_with_amentum': 'years_with_amentum', # Mentee's years
        'what_meeting_cadence_are_you_committed_to': 'meeting_cadence',
        'amentum_connection_networks': 'amentum_connection_networks', # Mentee's networks
        'why_are_you_interested_in_participating_in_the_mentoring_program': 'reason_for_participating',
        'did_you_previously_participate_in_an_amentum_or_jacobs_mentoring_program': 'previously_participated',
        'what_is_your_favorite_activityhobby': 'hobby', # Adjusted key based on likely standardization '/' removal
        'what_is_your_favorite_movie_genre': 'movie_genre',
        'what_is_your_favorite_book_genre': 'book_genre',
        'what_is_your_top_bucket_list_item': 'bucket_list', # Adjusted key based on likely standardization
        'what_is_a_fun_fact_about_you': 'fun_fact'
        # Add/modify mappings here if the debug print shows mismatches
    }
    # Only rename columns that actually exist to avoid errors
    existing_cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_cols_to_rename)

    # --- DEBUGGING PRINT ---
    print("\n--- Columns after final renaming: ---")
    print(df.columns.tolist())
    # --- END DEBUGGING PRINT ---

    return df

# --- Main Script ---

# 1. Load Data
print("Loading data...")
try:
    # Specify sheet name if necessary, e.g., pd.read_excel(MENTEE_FILE_PATH, sheet_name='Sheet1')
    df_mentees = pd.read_excel(MENTEE_FILE_PATH)
    df_mentors = pd.read_excel(MENTOR_FILE_PATH)
    print(f"Loaded {len(df_mentees)} mentees and {len(df_mentors)} mentors.")
except FileNotFoundError:
    print(f"Error: Could not find input files.")
    print(f"Please ensure '{MENTEE_FILE_PATH}' and '{MENTOR_FILE_PATH}' are correct.")
    exit()
except Exception as e:
    print(f"Error loading Excel files: {e}")
    exit()

# 2. Preprocessing and Standardization
print("\nPreprocessing data...")
# Use .copy() to avoid potential SettingWithCopyWarning later
df_mentees = standardize_col_names(df_mentees.copy())
df_mentors = standardize_col_names(df_mentors.copy())

# Ensure essential TARGET columns exist AFTER standardization/renaming
# Base required columns
required_mentee_base = ['id', 'name', 'manager']
required_mentor_base = ['id', 'name', 'mentor_capacity']

# Combine base and target semantic columns for the check
required_mentee_cols = list(set(required_mentee_base + MENTEE_TARGET_SEMANTIC_COLS))
required_mentor_cols = list(set(required_mentor_base + MENTOR_TARGET_SEMANTIC_COLS))

missing_mentee = [col for col in required_mentee_cols if col not in df_mentees.columns]
missing_mentor = [col for col in required_mentor_cols if col not in df_mentors.columns]

if missing_mentee:
    print(f"\nError: Missing required TARGET columns in Mentee data after standardization/renaming: {missing_mentee}")
    print("Compare this list with the 'Columns after final renaming' printout above.")
    print("You may need to adjust the 'rename_map' in the 'standardize_col_names' function.")
    print("Specifically, check the keys (left side) of the 'rename_map'. They should match the names")
    print("shown in the 'Columns after initial standardization' printout for the columns you want to rename.")
    exit()
if missing_mentor:
    print(f"\nError: Missing required TARGET columns in Mentor data after standardization/renaming: {missing_mentor}")
    print("Compare this list with the 'Columns after final renaming' printout above.")
    print("You may need to adjust the 'rename_map' in the 'standardize_col_names' function.")
    print("Specifically, check the keys (left side) of the 'rename_map'. They should match the names")
    print("shown in the 'Columns after initial standardization' printout for the columns you want to rename.")
    exit()

# Combine text fields for semantic analysis using TARGET columns
print("\nCombining text fields for semantic profiles...")
df_mentees['semantic_profile'] = df_mentees.apply(lambda row: combine_text_fields(row, MENTEE_TARGET_SEMANTIC_COLS), axis=1)
df_mentors['semantic_profile'] = df_mentors.apply(lambda row: combine_text_fields(row, MENTOR_TARGET_SEMANTIC_COLS), axis=1)

# Fill NaN capacity with a default (e.g., 1) or handle appropriately
df_mentors['mentor_capacity'] = df_mentors['mentor_capacity'].fillna(1).astype(int)

# Keep track of remaining mentor capacity
mentor_remaining_capacity = df_mentors.set_index('id')['mentor_capacity'].to_dict()

# 3. Semantic Embeddings
print("\nGenerating semantic embeddings (this may take a while)...")
# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

mentee_embeddings = model.encode(df_mentees['semantic_profile'].tolist(), show_progress_bar=True)
mentor_embeddings = model.encode(df_mentors['semantic_profile'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# 4. Matching Logic
print("\nCalculating matches...")
all_matches = []

# Calculate cosine similarity between all mentees and mentors
similarity_matrix = cosine_similarity(mentee_embeddings, mentor_embeddings)

# Convert mentor info to lists/arrays for easy lookup using TARGET names
mentor_ids = df_mentors['id'].tolist()
mentor_names = df_mentors['name'].tolist()
# Use .get(TARGET_COLUMN_NAME, default_series) for columns that might be optional or named differently
default_series_mentor = pd.Series(index=df_mentors.index) # Empty series as default
mentor_managers = df_mentors.get('manager', default_series_mentor).tolist()
mentor_cadence = df_mentors.get('meeting_cadence', default_series_mentor).tolist()
mentor_networks = df_mentors.get('amentum_connection_networks', default_series_mentor).tolist()
mentor_prefer_same_network = df_mentors.get('prefer_same_network_mentor', default_series_mentor).tolist()


for i, mentee in df_mentees.iterrows():
    mentee_id = mentee['id']
    mentee_name = mentee['name']
    # Use .get(TARGET_COLUMN_NAME, default_value) for potentially missing columns per row
    mentee_manager = mentee.get('manager', None)
    mentee_cadence = mentee.get('meeting_cadence', None)
    mentee_networks = mentee.get('amentum_connection_networks', None)
    mentee_prefer_same_network = mentee.get('prefer_same_network_mentee', None)

    # Get similarity scores for this mentee against all mentors
    scores = similarity_matrix[i]

    # Combine scores with mentor info
    mentor_scores = []
    for j, score in enumerate(scores):
        mentor_id = mentor_ids[j]
        mentor_name = mentor_names[j]
        mentor_cap = mentor_remaining_capacity.get(mentor_id, 0)

        # --- Apply Matching Constraints and Preferences ---
        final_score = score # Start with semantic similarity

        # Constraint: Mentor must have capacity
        if mentor_cap <= 0:
            continue

        # Constraint: Mentee cannot be mentored by their own manager
        # Ensure both mentee_manager and mentor_name are valid strings before comparing
        if pd.notna(mentee_manager) and isinstance(mentor_name, str) and mentor_name == mentee_manager:
             continue

        # Preference: Meeting Cadence
        # Ensure both cadences are valid strings before comparing
        if pd.notna(mentee_cadence) and pd.notna(mentor_cadence[j]) and mentee_cadence == mentor_cadence[j]:
            final_score += 0.1

        # Preference: Amentum Connection Networks
        # Standardize processing of network strings (handle NaN, split by ';', strip whitespace)
        mentee_nets_str = str(mentee_networks) if pd.notna(mentee_networks) else ''
        mentor_nets_str = str(mentor_networks[j]) if pd.notna(mentor_networks[j]) else ''
        mentee_nets = set(item.strip() for item in mentee_nets_str.split(';') if item.strip())
        mentor_nets = set(item.strip() for item in mentor_nets_str.split(';') if item.strip())
        common_nets = mentee_nets.intersection(mentor_nets)

        mentee_pref = str(mentee_prefer_same_network).lower() == 'yes'
        mentor_pref = str(mentor_prefer_same_network[j]).lower() == 'yes'

        if common_nets:
            if mentee_pref and mentor_pref:
                final_score += 0.15 # Strong boost
            elif mentee_pref or mentor_pref:
                 final_score += 0.05 # Smaller boost

        mentor_scores.append({
            'mentor_id': mentor_id,
            'mentor_name': mentor_name,
            'semantic_similarity': score,
            'final_score': final_score
        })

    # Sort potential mentors by the final score (descending)
    sorted_mentors = sorted(mentor_scores, key=lambda x: x['final_score'], reverse=True)

    # Store top N matches for the mentee (e.g., top 5)
    top_n = 5
    mentee_matches = {
        'mentee_id': mentee_id,
        'mentee_name': mentee_name,
    }
    for k in range(min(top_n, len(sorted_mentors))):
        match = sorted_mentors[k]
        mentee_matches[f'match_{k+1}_mentor_id'] = match['mentor_id']
        mentee_matches[f'match_{k+1}_mentor_name'] = match['mentor_name']
        mentee_matches[f'match_{k+1}_score'] = round(match['final_score'], 4)
        mentee_matches[f'match_{k+1}_semantic_similarity'] = round(match['semantic_similarity'], 4)

    all_matches.append(mentee_matches)

    # Optional Allocation Logic (remains commented out) ...

# 5. Output Results
print("\nSaving results...")
df_results = pd.DataFrame(all_matches)

# Reorder columns for clarity
cols_order = ['mentee_id', 'mentee_name']
for k in range(1, top_n + 1):
    # Check if match columns exist before adding to order
    if f'match_{k}_mentor_id' in df_results.columns:
        cols_order.extend([f'match_{k}_mentor_id', f'match_{k}_mentor_name', f'match_{k}_score', f'match_{k}_semantic_similarity'])
# Ensure only existing columns are included in the final order
df_results = df_results[[col for col in cols_order if col in df_results.columns]]

try:
    df_results.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"Matching suggestions saved to '{OUTPUT_FILE_PATH}'")
except Exception as e:
    print(f"Error saving results to Excel: {e}")

print("\nScript finished.")