import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re # For cleaning text

# --- Configuration ---
MENTEE_FILE_PATH = 'data/mentee_clean.xls' # <--- !!! UPDATE THIS PATH !!!
MENTOR_FILE_PATH = 'data/mentor_clean.xls' # <--- !!! UPDATE THIS PATH !!!
OUTPUT_FILE_PATH = 'mentor_mentee_matches.xlsx' # Path to save the results

# Columns to use for semantic matching (adjust as needed)
# Combine relevant free-text fields for a richer semantic profile
MENTEE_SEMANTIC_COLS = [
    'competencies you would like to work on with a mentor',
    'Why are you interested in participating in the Mentoring Program?',
    'What is your favorite activity/hobby?',
    'What is your favorite movie genre?',
    'What is your favorite book genre?',
    'What is a fun fact about you?'
]
MENTOR_SEMANTIC_COLS = [
    'Please select the competencies you feel you could teach',
    'Why are you interested in participating in the Mentoring Program?',
    'What is your favorite activity/hobby?',
    'What is your favorite movie genre?',
    'What is your favorite book genre?',
    'What is a fun fact about you?'
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
        if col in row and pd.notna(row[col]):
            combined.append(str(row[col]))
    return ". ".join(combined)

def standardize_col_names(df):
    """Standardize column names for easier access."""
    df.columns = [re.sub(r'\s+', '_', col.lower().strip().replace('?','').replace(':','')) for col in df.columns]
    # Manual renaming for known variations if necessary (example)
    df = df.rename(columns={
        'what_is_your_current_job_title': 'title',
        'select_your_occupation_category': 'category',
        'what_is_your_current_level_within_the_company': 'level',
        'who_is_your_direct_manager': 'manager',
        'in_which_time_zone_do_you_work_currently': 'time_zone',
        'how_many_years_have_you_been_with_amentum': 'years_with_amentum',
        'please_select_the_competencies_you_feel_you_could_teach': 'competencies_offered',
        'competencies_you_would_like_to_work_on_with_a_mentor': 'competencies_desired',
        'what_meeting_cadence_are_you_committed_to': 'meeting_cadence',
        'how_many_mentees_are_you_able_to_mentor_throughout_the_program': 'mentor_capacity',
        'what_amentum_connect_network(s)_are_you_a_member_of': 'amentum_connection_networks',
        'would_you_prefer_to_be_matched_with_a_mentor_in_the_same_amentum_connection_networks_as_you': 'prefer_same_network_mentee',
        'would_you_prefer_to_be_matched_with_a_mentee_in_the_same_amentum_connection_networks_as_you': 'prefer_same_network_mentor',
        'why_are_you_interested_in_participating_in_the_mentoring_program': 'reason_for_participating',
        'did_you_previously_participate_in_an_amentum_or_jacobs_mentoring_program': 'previously_participated',
        'what_is_your_favorite_activity/hobby': 'hobby',
        'what_is_your_favorite_movie_genre': 'movie_genre',
        'what_is_your_favorite_book_genre': 'book_genre',
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mixter_of_both': 'personality', # Note: 'mixter' in original mentor q21
        'do_you_consider_yourself_an_introvert_extrovert_or_a_mix_of_both': 'personality',
        'what_is_your_top_"bucket_list"_item': 'bucket_list',
        'what_is_a_fun_fact_about_you': 'fun_fact'
    })
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
print("Preprocessing data...")
df_mentees = standardize_col_names(df_mentees)
df_mentors = standardize_col_names(df_mentors)

# Ensure essential columns exist
required_mentee_cols = ['id', 'name', 'manager'] + [col.replace(' ', '_').lower() for col in MENTEE_SEMANTIC_COLS]
required_mentor_cols = ['id', 'name', 'mentor_capacity'] + [col.replace(' ', '_').lower() for col in MENTOR_SEMANTIC_COLS]

missing_mentee = [col for col in required_mentee_cols if col not in df_mentees.columns]
missing_mentor = [col for col in required_mentor_cols if col not in df_mentors.columns]

if missing_mentee:
    print(f"Error: Missing required columns in Mentee data: {missing_mentee}")
    exit()
if missing_mentor:
    print(f"Error: Missing required columns in Mentor data: {missing_mentor}")
    exit()

# Standardize semantic columns used in the combine function
mentee_semantic_cols_std = [col.replace(' ', '_').lower() for col in MENTEE_SEMANTIC_COLS]
mentor_semantic_cols_std = [col.replace(' ', '_').lower() for col in MENTOR_SEMANTIC_COLS]

# Combine text fields for semantic analysis
df_mentees['semantic_profile'] = df_mentees.apply(lambda row: combine_text_fields(row, mentee_semantic_cols_std), axis=1)
df_mentors['semantic_profile'] = df_mentors.apply(lambda row: combine_text_fields(row, mentor_semantic_cols_std), axis=1)

# Fill NaN capacity with a default (e.g., 1) or handle appropriately
df_mentors['mentor_capacity'] = df_mentors['mentor_capacity'].fillna(1).astype(int)

# Keep track of remaining mentor capacity
mentor_remaining_capacity = df_mentors.set_index('id')['mentor_capacity'].to_dict()

# 3. Semantic Embeddings
print("Generating semantic embeddings (this may take a while)...")
# Load a pre-trained sentence transformer model
# 'all-MiniLM-L6-v2' is a good balance of speed and performance
# Other options: 'paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2' (better but slower)
model = SentenceTransformer('all-MiniLM-L6-v2')

mentee_embeddings = model.encode(df_mentees['semantic_profile'].tolist(), show_progress_bar=True)
mentor_embeddings = model.encode(df_mentors['semantic_profile'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# 4. Matching Logic
print("Calculating matches...")
all_matches = []

# Calculate cosine similarity between all mentees and mentors
similarity_matrix = cosine_similarity(mentee_embeddings, mentor_embeddings)

# Convert mentor IDs and names to lists/arrays for easy lookup
mentor_ids = df_mentors['id'].tolist()
mentor_names = df_mentors['name'].tolist()
mentor_managers = df_mentors.get('manager', pd.Series(index=df_mentors.index, name='manager')).tolist() # Handle potential missing manager col
mentor_cadence = df_mentors.get('meeting_cadence', pd.Series(index=df_mentors.index, name='meeting_cadence')).tolist()
mentor_networks = df_mentors.get('amentum_connection_networks', pd.Series(index=df_mentors.index, name='amentum_connection_networks')).tolist()
mentor_prefer_same_network = df_mentors.get('prefer_same_network_mentor', pd.Series(index=df_mentors.index, name='prefer_same_network_mentor')).tolist()


for i, mentee in df_mentees.iterrows():
    mentee_id = mentee['id']
    mentee_name = mentee['name']
    mentee_manager = mentee.get('manager', None) # Handle potential missing manager col
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
            # print(f"Skipping {mentor_name} for {mentee_name} (no capacity)")
            continue # Skip this mentor if they have no capacity left

        # Constraint: Mentee cannot be mentored by their own manager (if manager info exists)
        # This checks if the mentor IS the mentee's manager.
        if mentee_manager is not None and pd.notna(mentee_manager) and mentor_name == mentee_manager:
             # print(f"Skipping {mentor_name} for {mentee_name} (mentor is manager)")
             continue # Skip if mentor is the mentee's direct manager

        # Constraint: Mentor cannot be the Mentee's manager's manager (Optional - add if needed)
        # mentor_manager = mentor_managers[j]
        # if mentee_manager is not None and mentor_manager is not None and pd.notna(mentee_manager) and pd.notna(mentor_manager) and mentee_manager == mentor_manager:
        #     continue # Skip if mentor and mentee share the same manager

        # Preference: Meeting Cadence (Exact match boost, adjust scoring as needed)
        if mentee_cadence and mentor_cadence[j] and mentee_cadence == mentor_cadence[j]:
            final_score += 0.1 # Add a bonus for matching cadence

        # Preference: Amentum Connection Networks
        mentee_nets = set(str(mentee_networks).split(';')) if pd.notna(mentee_networks) else set()
        mentor_nets = set(str(mentor_networks[j]).split(';')) if pd.notna(mentor_networks[j]) else set()
        common_nets = mentee_nets.intersection(mentor_nets)

        # Boost if both prefer same network and they share one
        if common_nets:
            if mentee_prefer_same_network == 'Yes' and mentor_prefer_same_network[j] == 'Yes':
                final_score += 0.15 # Strong boost
            elif mentee_prefer_same_network == 'Yes' or mentor_prefer_same_network[j] == 'Yes':
                 final_score += 0.05 # Smaller boost if only one prefers it

        # --- Add other potential factors ---
        # Example: Level difference (prefer mentor higher level) - Requires 'level' column standardization
        # Example: Competency keyword matching (in addition to semantic)

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

    # --- Simple Allocation (Optional - First Best Match) ---
    # This is a basic greedy allocation. More sophisticated methods exist (e.g., Stable Matching).
    # If you uncomment this, it will assign the top available mentor and decrement capacity.
    # allocated = False
    # for match in sorted_mentors:
    #     mentor_id_to_assign = match['mentor_id']
    #     if mentor_remaining_capacity.get(mentor_id_to_assign, 0) > 0:
    #         print(f"Assigning {mentee_name} to {match['mentor_name']}")
    #         mentor_remaining_capacity[mentor_id_to_assign] -= 1
    #         # Add assignment details to a separate list/df if needed
    #         allocated = True
    #         break
    # if not allocated and sorted_mentors:
    #     print(f"Could not assign {mentee_name} - no suitable mentors with capacity found.")
    # elif not sorted_mentors:
    #      print(f"Could not assign {mentee_name} - no suitable mentors found after filtering.")


# 5. Output Results
print("Saving results...")
df_results = pd.DataFrame(all_matches)

# Reorder columns for clarity
cols_order = ['mentee_id', 'mentee_name']
for k in range(1, top_n + 1):
    cols_order.extend([f'match_{k}_mentor_id', f'match_{k}_mentor_name', f'match_{k}_score', f'match_{k}_semantic_similarity'])
# Ensure only existing columns are included in the final order
df_results = df_results[[col for col in cols_order if col in df_results.columns]]

try:
    df_results.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"Matching suggestions saved to '{OUTPUT_FILE_PATH}'")
except Exception as e:
    print(f"Error saving results to Excel: {e}")

print("Script finished.")
