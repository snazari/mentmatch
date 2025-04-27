import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # For cleaning text

# --- Configuration ---
MENTEE_FILE_PATH = '~/sandbox/mentmatch/data/mentee_clean.xlsx' # <--- !!! UPDATE THIS PATH !!!
MENTOR_FILE_PATH = '~/sandbox/mentmatch/data/mentor_clean_v2.xlsx' # <--- !!! UPDATE THIS PATH !!!
OUTPUT_FILE_PATH = '~/sandbox/mentmatch/mentor_mentee_matches_20250427_new_weighted.xlsx' # Path to save the results

# --- NEW: Define columns for Career vs Social Profiles ---
# Adjust these lists based on the final standardized column names
# Ensure these columns actually exist after standardization
CAREER_COLS_MENTEE = ['competencies_desired', 'title', 'category', 'level']
SOCIAL_COLS_MENTEE = ['reason_for_participating', 'hobby', 'movie_genre', 'book_genre', 'fun_fact']
CAREER_COLS_MENTOR = ['competencies_offered', 'title', 'category', 'level']
SOCIAL_COLS_MENTOR = ['reason_for_participating', 'hobby', 'movie_genre', 'book_genre', 'fun_fact']

# --- NEW: Define Weights for Career vs Social Score ---
CAREER_WEIGHT = 0.7
SOCIAL_WEIGHT = 0.3

# Define the TARGET standardized column names we expect to use after renaming
# These correspond to the values in the .rename() dictionary keys used for semantic profiles
# Combine career and social for checking required columns
MENTEE_TARGET_SEMANTIC_COLS = list(set(CAREER_COLS_MENTEE + SOCIAL_COLS_MENTEE))
MENTOR_TARGET_SEMANTIC_COLS = list(set(CAREER_COLS_MENTOR + SOCIAL_COLS_MENTOR))

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
# Create separate profiles
print(" - Generating Career Profiles")
df_mentees['career_profile'] = df_mentees.apply(lambda row: combine_text_fields(row, CAREER_COLS_MENTEE), axis=1)
df_mentors['career_profile'] = df_mentors.apply(lambda row: combine_text_fields(row, CAREER_COLS_MENTOR), axis=1)
print(" - Generating Social Profiles")
df_mentees['social_profile'] = df_mentees.apply(lambda row: combine_text_fields(row, SOCIAL_COLS_MENTEE), axis=1)
df_mentors['social_profile'] = df_mentors.apply(lambda row: combine_text_fields(row, SOCIAL_COLS_MENTOR), axis=1)


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
# Create a mutable copy for decrementing during assignment
if 'id' in df_mentors.columns:
    if df_mentors['id'].is_unique:
        mentor_capacity_map = df_mentors.set_index('id')['mentor_capacity'].to_dict()
    else:
        # Handle duplicate mentor IDs if necessary (as before)
        print("Warning: Duplicate mentor IDs found. Using capacity from the first occurrence of each ID.")
        mentor_capacity_series = df_mentors.drop_duplicates(subset=['id']).set_index('id')['mentor_capacity']
        mentor_capacity_map = mentor_capacity_series.to_dict()
else:
    print("Error: 'id' column not found in mentor data after renaming. Cannot track capacity.")
    exit()


# 3. Semantic Embeddings
print("\nGenerating semantic embeddings (this may take a while)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
# Generate Career Embeddings
print(" - Encoding Career Profiles...")
mentee_career_embeddings = model.encode(df_mentees['career_profile'].tolist(), show_progress_bar=True)
mentor_career_embeddings = model.encode(df_mentors['career_profile'].tolist(), show_progress_bar=True)
# Generate Social Embeddings
print(" - Encoding Social Profiles...")
mentee_social_embeddings = model.encode(df_mentees['social_profile'].tolist(), show_progress_bar=True)
mentor_social_embeddings = model.encode(df_mentors['social_profile'].tolist(), show_progress_bar=True)

# Drop profile columns to save memory if needed
# df_mentees = df_mentees.drop(columns=['career_profile', 'social_profile'])
# df_mentors = df_mentors.drop(columns=['career_profile', 'social_profile'])

print("Embeddings generated.")

# 4. Matching Logic - Greedy Assignment with Weighted Score
print("\nCalculating all potential matches and weighted scores...")
# Calculate separate similarity matrices
career_similarity_matrix = cosine_similarity(mentee_career_embeddings, mentor_career_embeddings)
social_similarity_matrix = cosine_similarity(mentee_social_embeddings, mentor_social_embeddings)

# Convert mentor info to lists/arrays for faster lookup during score calculation
mentor_ids = df_mentors['id'].tolist()
mentor_names = df_mentors['name'].tolist()
default_series_mentor = pd.Series(index=df_mentors.index) # Empty series for safe .get()
mentor_managers = df_mentors.get('manager', default_series_mentor).tolist()
mentor_cadence = df_mentors.get('meeting_cadence', default_series_mentor).tolist()
mentor_networks = df_mentors.get('amentum_connection_networks', default_series_mentor).tolist()
mentor_prefer_same_network = df_mentors.get('prefer_same_network_mentor', default_series_mentor).tolist()
mentor_competencies_offered = df_mentors.get('competencies_offered', default_series_mentor).tolist()
mentor_hobbies = df_mentors.get('hobby', default_series_mentor).tolist()
mentor_movie_genres = df_mentors.get('movie_genre', default_series_mentor).tolist()
mentor_book_genres = df_mentors.get('book_genre', default_series_mentor).tolist()


potential_assignments = []

for i, mentee in df_mentees.iterrows():
    mentee_id = mentee['id']
    mentee_name = mentee['name']
    mentee_manager = mentee.get('manager', None)
    mentee_cadence = mentee.get('meeting_cadence', None)
    mentee_networks = mentee.get('amentum_connection_networks', None)
    mentee_prefer_same_network = mentee.get('prefer_same_network_mentee', None)
    mentee_competencies = str(mentee.get('competencies_desired', ''))
    mentee_hobby = str(mentee.get('hobby', ''))
    mentee_movie_genre = str(mentee.get('movie_genre', ''))
    mentee_book_genre = str(mentee.get('book_genre', ''))

    # Get career and social similarity scores for this mentee against all mentors
    mentee_career_scores = career_similarity_matrix[i]
    mentee_social_scores = social_similarity_matrix[i]
    
    # Iterate through mentors to calculate combined scores
    for j, mentor_id in enumerate(mentor_ids): # Use enumerate over mentor_ids for index j
        # mentor_id = mentor_ids[j] # Already got mentor_id from enumerate
        mentor_name = mentor_names[j]

        # --- Initial Check: Skip if mentor is mentee's manager ---
        if pd.notna(mentee_manager) and isinstance(mentor_name, str) and mentor_name == mentee_manager:
            continue

        # --- Calculate Weighted Semantic Score ---
        career_score = mentee_career_scores[j]
        social_score = mentee_social_scores[j]
        weighted_semantic_score = (CAREER_WEIGHT * career_score) + (SOCIAL_WEIGHT * social_score)

        # --- Calculate Final Score with Adjustments ---
        final_score = weighted_semantic_score # Start with weighted semantic score
        match_reasons = []
        # Add breakdown of semantic scores to reasons
        match_reasons.append(f"Weighted Semantic: {weighted_semantic_score:.2f} (Career: {career_score:.2f}, Social: {social_score:.2f})")

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
        mentor_competencies = str(mentor_competencies_offered[j])
        if mentee_competencies and mentor_competencies:
            match_reasons.append(f"Competency match: mentee seeks '{mentee_competencies.strip()}', mentor offers '{mentor_competencies.strip()}'")

        # Other match factors - add shared interests
        mentor_hobby = str(mentor_hobbies[j])
        if mentee_hobby and mentor_hobby and mentee_hobby.lower() == mentor_hobby.lower():
            match_reasons.append(f"Shared hobby: {mentee_hobby}")
        mentor_movie = str(mentor_movie_genres[j])
        if mentee_movie_genre and mentor_movie and mentee_movie_genre.lower() == mentor_movie.lower():
             match_reasons.append(f"Shared movie genre: {mentee_movie_genre}")
        mentor_book = str(mentor_book_genres[j])
        if mentee_book_genre and mentor_book and mentee_book_genre.lower() == mentor_book.lower():
             match_reasons.append(f"Shared book genre: {mentee_book_genre}")

        # Create a concise match summary
        match_summary = "; ".join(match_reasons)

        potential_assignments.append({
            'mentee_idx': i,
            'mentor_idx': j,
            'mentee_id': mentee_id,
            'mentor_id': mentor_id,
            'mentee_name': mentee_name,
            'mentor_name': mentor_name,
            # Store individual scores for potential analysis
            'career_similarity': career_score,
            'social_similarity': social_score,
            'weighted_semantic_score': weighted_semantic_score,
            'final_score': final_score,
            'match_summary': match_summary
        })

# Sort all potential assignments by final score, descending
potential_assignments.sort(key=lambda x: x['final_score'], reverse=True)
print(f"Calculated {len(potential_assignments)} potential assignments.")

# Perform greedy assignment
print("\nPerforming greedy assignment...")
final_matches = []
assigned_mentees = set() # Keep track of mentees who have been assigned

# Use the mentor_capacity_map which will be decremented
current_mentor_capacity = mentor_capacity_map.copy()

for assignment in potential_assignments:
    mentee_id = assignment['mentee_id']
    mentor_id = assignment['mentor_id']

    # Check if mentee is already assigned OR if mentor has capacity
    if mentee_id not in assigned_mentees and current_mentor_capacity.get(mentor_id, 0) > 0:
        # Assign this match
        final_matches.append({
            'mentee_id': mentee_id,
            'mentee_name': assignment['mentee_name'],
            'assigned_mentor_id': mentor_id,
            'assigned_mentor_name': assignment['mentor_name'],
            'match_score': round(assignment['final_score'], 4),
            # Include separate scores in output
            'career_similarity': round(assignment['career_similarity'], 4),
            'social_similarity': round(assignment['social_similarity'], 4),
            'weighted_semantic_score': round(assignment['weighted_semantic_score'], 4),
            'match_summary': assignment['match_summary']
        })

        # Update tracking
        assigned_mentees.add(mentee_id)
        current_mentor_capacity[mentor_id] -= 1

print(f"Assigned {len(final_matches)} mentees.")

# Check for unassigned mentees
unassigned_mentee_ids = set(df_mentees['id']) - assigned_mentees
if unassigned_mentee_ids:
    print(f"Warning: {len(unassigned_mentee_ids)} mentees could not be assigned (likely due to mentor capacity constraints).")
    
    # --- NEW: Save unassigned mentees to a separate file ---
    print("\nSaving list of unassigned mentees...")
    try:
        # Define the path for the unassigned mentees file
        unassigned_file_path = OUTPUT_FILE_PATH.replace('.xlsx', '_unassigned.xlsx')
        
        # Filter the original mentee DataFrame
        df_unassigned = df_mentees[df_mentees['id'].isin(unassigned_mentee_ids)].copy()
        
        # Select relevant columns to save (adjust as needed)
        unassigned_cols = ['id', 'name', 'email', 'manager', 'title', 'level', 'category'] # Basic info
        # Add profile columns if they weren't dropped earlier
        if 'career_profile' in df_unassigned.columns:
             unassigned_cols.append('career_profile')
        if 'social_profile' in df_unassigned.columns:
             unassigned_cols.append('social_profile')
        # Add original semantic columns used for matching
        unassigned_cols.extend([col for col in MENTEE_TARGET_SEMANTIC_COLS if col in df_unassigned.columns])
        
        # Keep only existing columns from the desired list
        df_unassigned_output = df_unassigned[[col for col in unassigned_cols if col in df_unassigned.columns]]
        
        df_unassigned_output.to_excel(unassigned_file_path, index=False)
        print(f"Unassigned mentees saved to '{unassigned_file_path}'")
    except Exception as e:
        print(f"Error saving unassigned mentees list: {e}")
    # --- END NEW SECTION ---
    
    # Optionally, list them or save to a separate file

# 5. Output Results
print("\nSaving results...")
df_results = pd.DataFrame(final_matches)

# Define the order of columns for the output file
cols_order = [
    'mentee_id',
    'mentee_name',
    'assigned_mentor_id',
    'assigned_mentor_name',
    'match_score',
    # Add new score columns to output
    'weighted_semantic_score',
    'career_similarity',
    'social_similarity',
    'match_summary'
]

# Ensure all columns exist before reordering (should exist based on creation)
df_results = df_results[[col for col in cols_order if col in df_results.columns]]

try:
    df_results.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"Final assignments saved to '{OUTPUT_FILE_PATH}'")
except Exception as e:
    print(f"Error saving results to Excel: {e}")

print("\nScript finished.")
