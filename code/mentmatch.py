import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # For cleaning text

# --- Configuration ---
MENTEE_FILE_PATH = '/datadrive/part1/sandbox/mentmatch/data/mentee_clean.xlsx' # <--- !!! UPDATE THIS PATH !!!
MENTOR_FILE_PATH = '/datadrive/part1/sandbox/mentmatch/data/mentor_clean_v2.xlsx' # <--- !!! UPDATE THIS PATH !!!
OUTPUT_FILE_PATH = '/datadrive/part1/sandbox/mentmatch/mentor_mentee_matches_20250415.xlsx' # Path to save the results

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
    return ". ".join(combined)


def standardize_col_names(df):
    """Standardize column names for easier access."""
    # Step 1: Basic standardization
    original_columns = df.columns.tolist()
    standardized_columns = [re.sub(r'[?"\':()\[\];,./]', '', col.lower().strip()) for col in original_columns]
    standardized_columns = [re.sub(r'\s+', '_', col) for col in standardized_columns]
    standardized_columns = [re.sub(r'_+', '_', col) for col in standardized_columns]
    df.columns = standardized_columns

    # --- DEBUGGING PRINT ---
    print("\n--- Columns after initial standardization (before renaming): ---")
    print(df.columns.tolist())
    # --- END DEBUGGING PRINT ---

    # Step 2: Manual renaming
    rename_map = {
        # Mentee specific
        'competencies_you_would_like_to_work_on_with_a_mentor': 'competencies_desired',
        'would_you_prefer_to_be_matched_with_a_mentor_in_the_same_amentum_connection_networks_as_you': 'prefer_same_network_mentee',
         # Handle specific mentee variation if different from mentor's standardized name
        'would_you_prefer_to_be_matched_with_a_mentor_in_the_same_acns_as_you': 'prefer_same_network_mentee', # Added based on user output
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mix_of_both': 'personality', # Mentee version
        # Mentor specific
        'what_is_your_current_job_title': 'title',
        'select_your_occupation_category': 'category',
        'what_is_your_current_level_within_the_company': 'level',
        'who_is_your_direct_manager': 'manager', # Mentor's manager
        'in_which_time_zone_do_you_work_currently': 'time_zone',
        'how_many_years_have_you_been_with_amentum': 'years_with_amentum',
        # Use the exact name from Mentor's "initial standardization" printout
        'as_a_mentor_you_will_be_helping_a_mentee_develop_core_competencies_that_will_enable_them_to_become_future_leaders_within_the_organization_please_select_the_competencies_you_feel_you_could_teach_': 'competencies_offered', # Adjusted based on user output
        'how_many_mentees_are_you_able_to_mentor_throughout_the_program': 'mentor_capacity',
        'what_amentum_connect_networks_are_you_a_member_of': 'amentum_connection_networks',
         # Handle specific mentor variation if different from mentee's standardized name
        'would_you_prefer_to_be_matched_with_a_mentee_in_the_same_acns_as_you': 'prefer_same_network_mentor', # Added based on user output
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mixter_of_both': 'personality', # Mentor version ('mixter')
         # Common columns mapped to target names
        'name': 'name',
        'id': 'id',
        'email': 'email',
        'manager': 'manager',
        'title': 'title',
        'level': 'level',
        'category': 'category',
        'time_zone': 'time_zone',
        'years_with_amentum': 'years_with_amentum',
        'what_meeting_cadence_are_you_committed_to': 'meeting_cadence',
        'amentum_connection_networks': 'amentum_connection_networks',
        'why_are_you_interested_in_participating_in_the_mentoring_program': 'reason_for_participating',
        'did_you_previously_participate_in_an_amentum_or_jacobs_mentoring_program': 'previously_participated',
        'what_is_your_favorite_activityhobby': 'hobby',
        'what_is_your_favorite_movie_genre': 'movie_genre',
        'what_is_your_favorite_book_genre': 'book_genre',
        'what_is_your_top_bucket_list_item': 'bucket_list',
        'what_is_a_fun_fact_about_you': 'fun_fact'
    }
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
df_mentees = standardize_col_names(df_mentees.copy())
df_mentors = standardize_col_names(df_mentors.copy())

# Ensure essential TARGET columns exist AFTER standardization/renaming
required_mentee_base = ['id', 'name', 'manager']
required_mentor_base = ['id', 'name', 'mentor_capacity']
required_mentee_cols = list(set(required_mentee_base + MENTEE_TARGET_SEMANTIC_COLS))
required_mentor_cols = list(set(required_mentor_base + MENTOR_TARGET_SEMANTIC_COLS))

missing_mentee = [col for col in required_mentee_cols if col not in df_mentees.columns]
missing_mentor = [col for col in required_mentor_cols if col not in df_mentors.columns]

if missing_mentee:
    print(f"\nError: Missing required TARGET columns in Mentee data: {missing_mentee}")
    # ... (error message details) ...
    exit()
if missing_mentor:
    print(f"\nError: Missing required TARGET columns in Mentor data: {missing_mentor}")
    # ... (error message details) ...
    exit()

# Combine text fields for semantic analysis using TARGET columns
print("\nCombining text fields for semantic profiles...")
df_mentees['semantic_profile'] = df_mentees.apply(lambda row: combine_text_fields(row, MENTEE_TARGET_SEMANTIC_COLS), axis=1)
df_mentors['semantic_profile'] = df_mentors.apply(lambda row: combine_text_fields(row, MENTOR_TARGET_SEMANTIC_COLS), axis=1)


# --- Data Cleaning for Mentor Capacity ---
print("\nCleaning mentor capacity data...")
if 'mentor_capacity' in df_mentors.columns:
    # Define mapping for common text numbers (add more if needed)
    num_map_lower = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    # Create a map that includes original case and potential numeric strings
    num_map_combined = {k: v for k, v in num_map_lower.items()}
    num_map_combined.update({k.capitalize(): v for k, v in num_map_lower.items()}) # Add capitalized
    num_map_combined.update({str(i): i for i in range(1, 11)}) # Map numeric strings '1'->1 etc.

    # Apply replacement using the map
    # Convert column to string first to handle mixed types robustly before replacing
    # Use a temporary column to store intermediate string representation
    df_mentors['mentor_capacity_str'] = df_mentors['mentor_capacity'].astype(str).str.strip()
    df_mentors['mentor_capacity_cleaned'] = df_mentors['mentor_capacity_str'].replace(num_map_combined)

    # Convert to numeric, coercing errors (values not in map and not numeric) to NaN
    df_mentors['mentor_capacity_numeric'] = pd.to_numeric(df_mentors['mentor_capacity_cleaned'], errors='coerce')

    # Identify original values that failed conversion (for debugging)
    failed_conversion_mask = df_mentors['mentor_capacity_numeric'].isna() & df_mentors['mentor_capacity'].notna()
    if failed_conversion_mask.any():
        failed_values = df_mentors.loc[failed_conversion_mask, 'mentor_capacity'].unique()
        print(f"Warning: Could not convert the following original mentor capacities to numbers: {failed_values.tolist()}")
        print("These will be treated as NaN and filled with 1.")

    # Fill NaN capacity with a default (e.g., 1) and convert to integer
    # Use the numeric column for filling and conversion
    df_mentors['mentor_capacity'] = df_mentors['mentor_capacity_numeric'].fillna(1).astype(int)
    print("Mentor capacity cleaned and converted to integer.")

    # Optional: Drop intermediate columns if desired
    df_mentors = df_mentors.drop(columns=['mentor_capacity_str', 'mentor_capacity_cleaned', 'mentor_capacity_numeric'])

else:
    # This case should not happen based on previous checks, but good practice
    print("Error: 'mentor_capacity' column not found before cleaning step. Cannot proceed.")
    exit()
# --- END Data Cleaning Section ---


# Keep track of remaining mentor capacity using the correct index ('id')
if 'id' in df_mentors.columns:
    if df_mentors['id'].is_unique:
        mentor_remaining_capacity = df_mentors.set_index('id')['mentor_capacity'].to_dict()
    else:
        # Handle duplicate mentor IDs if necessary (as before)
        print("Warning: Duplicate mentor IDs found. Using capacity from the first occurrence of each ID.")
        mentor_capacity_series = df_mentors.drop_duplicates(subset=['id']).set_index('id')['mentor_capacity']
        mentor_remaining_capacity = mentor_capacity_series.to_dict()
else:
    print("Error: 'id' column not found in mentor data after renaming. Cannot track capacity.")
    exit()


# 3. Semantic Embeddings
print("\nGenerating semantic embeddings (this may take a while)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
mentee_embeddings = model.encode(df_mentees['semantic_profile'].tolist(), show_progress_bar=True)
mentor_embeddings = model.encode(df_mentors['semantic_profile'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# 4. Matching Logic
print("\nCalculating matches...")
all_matches = []
similarity_matrix = cosine_similarity(mentee_embeddings, mentor_embeddings)

# Convert mentor info to lists/arrays
mentor_ids = df_mentors['id'].tolist()
mentor_names = df_mentors['name'].tolist()
default_series_mentor = pd.Series(index=df_mentors.index)
mentor_managers = df_mentors.get('manager', default_series_mentor).tolist()
mentor_cadence = df_mentors.get('meeting_cadence', default_series_mentor).tolist()
mentor_networks = df_mentors.get('amentum_connection_networks', default_series_mentor).tolist()
mentor_prefer_same_network = df_mentors.get('prefer_same_network_mentor', default_series_mentor).tolist()

# Store additional data for match summary
mentor_df_index = {id: idx for idx, id in enumerate(mentor_ids)}


for i, mentee in df_mentees.iterrows():
    mentee_id = mentee['id']
    mentee_name = mentee['name']
    mentee_manager = mentee.get('manager', None)
    mentee_cadence = mentee.get('meeting_cadence', None)
    mentee_networks = mentee.get('amentum_connection_networks', None)
    mentee_prefer_same_network = mentee.get('prefer_same_network_mentee', None)

    scores = similarity_matrix[i]
    mentor_scores = []
    for j, score in enumerate(scores):
        mentor_id = mentor_ids[j]
        mentor_name = mentor_names[j]
        mentor_cap = mentor_remaining_capacity.get(mentor_id, 0) # Use tracking dict

        # --- Apply Matching Constraints and Preferences ---
        final_score = score
        match_reasons = []
        match_reasons.append(f"Semantic similarity: {score:.2f}")

        # Skip if mentor capacity is exhausted or mentor is mentee's manager
        if mentor_cap <= 0: continue
        if pd.notna(mentee_manager) and isinstance(mentor_name, str) and mentor_name == mentee_manager: continue

        # Cadence Preference
        mentor_cad = mentor_cadence[j]
        if pd.notna(mentee_cadence) and pd.notna(mentor_cad) and mentee_cadence == mentor_cad:
            final_score += 0.1
            match_reasons.append(f"Meeting cadence match: {mentor_cad}")

        # Network Preference
        mentee_nets_str = str(mentee_networks) if pd.notna(mentee_networks) else ''
        mentor_nets_str = str(mentor_networks[j]) if pd.notna(mentor_networks[j]) else ''
        mentee_nets = set(item.strip() for item in mentee_nets_str.split(';') if item.strip())
        mentor_nets = set(item.strip() for item in mentor_nets_str.split(';') if item.strip())
        common_nets = mentee_nets.intersection(mentor_nets)
        mentee_pref = str(mentee_prefer_same_network).lower() == 'yes'
        mentor_pref = str(mentor_prefer_same_network[j]).lower() == 'yes'
        if common_nets:
            if mentee_pref and mentor_pref:
                final_score += 0.15
                match_reasons.append(f"Both prefer same network: {', '.join(common_nets)}")
            elif mentee_pref or mentor_pref:
                final_score += 0.05
                match_reasons.append(f"One prefers same network: {', '.join(common_nets)}")
            else:
                match_reasons.append(f"Common networks: {', '.join(common_nets)}")

        # Competency match (add to summary)
        mentee_competencies = str(mentee.get('competencies_desired', ''))
        mentor_competencies = str(df_mentors.iloc[j].get('competencies_offered', ''))
        if mentee_competencies and mentor_competencies:
            match_reasons.append(f"Competency match: mentee seeks {mentee_competencies.strip()}, mentor offers {mentor_competencies.strip()}")

        # Other match factors - add shared interests
        for interest_field in ['hobby', 'movie_genre', 'book_genre']:
            mentee_interest = str(mentee.get(interest_field, ''))
            mentor_interest = str(df_mentors.iloc[j].get(interest_field, ''))
            if mentee_interest and mentor_interest and mentee_interest.lower() == mentor_interest.lower():
                match_reasons.append(f"Shared {interest_field.replace('_', ' ')}: {mentee_interest}")

        # Create a concise match summary
        match_summary = "; ".join(match_reasons)

        mentor_scores.append({
            'mentor_id': mentor_id, 'mentor_name': mentor_name,
            'semantic_similarity': score, 'final_score': final_score,
            'match_summary': match_summary
        })

    # Sort and store top N matches
    sorted_mentors = sorted(mentor_scores, key=lambda x: x['final_score'], reverse=True)
    top_n = 5
    mentee_matches = {'mentee_id': mentee_id, 'mentee_name': mentee_name}
    for k in range(min(top_n, len(sorted_mentors))):
        match = sorted_mentors[k]
        mentee_matches[f'match_{k+1}_mentor_id'] = match['mentor_id']
        mentee_matches[f'match_{k+1}_mentor_name'] = match['mentor_name']
        mentee_matches[f'match_{k+1}_score'] = round(match['final_score'], 4)
        mentee_matches[f'match_{k+1}_semantic_similarity'] = round(match['semantic_similarity'], 4)
        mentee_matches[f'match_{k+1}_summary'] = match['match_summary']
    all_matches.append(mentee_matches)

# 5. Output Results
print("\nSaving results...")
df_results = pd.DataFrame(all_matches)
cols_order = ['mentee_id', 'mentee_name']
for k in range(1, top_n + 1):
    if f'match_{k}_mentor_id' in df_results.columns:
        cols_order.extend([
            f'match_{k}_mentor_id',
            f'match_{k}_mentor_name',
            f'match_{k}_score',
            f'match_{k}_semantic_similarity',
            f'match_{k}_summary'
        ])
df_results = df_results[[col for col in cols_order if col in df_results.columns]]

try:
    df_results.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"Matching suggestions saved to '{OUTPUT_FILE_PATH}'")
except Exception as e:
    print(f"Error saving results to Excel: {e}")

print("\nScript finished.")
