# MentMatch: Semantic Mentorship Matching

This Python script facilitates matching mentees with suitable mentors based on their self-described competencies, interests, and goals using semantic similarity. It leverages sentence embeddings to understand the nuances in text descriptions provided in input Excel files.

## Workflow Overview

The script performs the following steps:

1.  **Load Data**: Reads mentee and mentor information from specified Excel files (`mentee_clean.xlsx` and `mentor_clean.xlsx` by default, located in the `./data/` directory).
2.  **Preprocessing & Standardization**:
    *   Cleans and standardizes column names from the input files for consistency. This involves removing special characters, converting to lowercase, replacing spaces with underscores, and applying a manual renaming map for specific columns crucial for the matching logic.
    *   Checks for the existence of essential columns after standardization.
    *   Combines relevant text fields (like desired/offered competencies, reasons for participating, hobbies, etc.) into a single "semantic profile" string for both mentees and mentors. These are defined in `MENTEE_TARGET_SEMANTIC_COLS` and `MENTOR_TARGET_SEMANTIC_COLS`.
    *   Cleans the 'mentor\_capacity' column, converting text representations of numbers (e.g., "one", "Two") into integers and handling potential errors or missing values.
3.  **Semantic Embedding Generation**:
    *   Uses a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) to convert the combined `semantic_profile` text for each mentee and mentor into dense vector representations (embeddings). These embeddings capture the semantic meaning of the text.
4.  **Similarity Calculation**:
    *   Calculates the cosine similarity between the embedding of each mentee and the embedding of every mentor. Cosine similarity measures the angular difference between vectors, providing a score between -1 and 1 (or 0 and 1 in this context, as embeddings are non-negative), where higher values indicate greater semantic similarity.
5.  **Matching Algorithm**:
    *   Iterates through each mentee.
    *   For each mentee, sorts potential mentors based on their calculated semantic similarity score in descending order.
    *   Filters out mentors who are the mentee's direct manager (if manager information is available).
    *   Selects the highest-scoring available mentor who still has capacity (based on the cleaned `mentor_capacity` column).
    *   Assigns the mentee to the selected mentor and decrements that mentor's remaining capacity.
    *   If no suitable mentor is found (e.g., all high-similarity mentors are managers or at capacity), the mentee remains unassigned.
6.  **Output**:
    *   Creates a Pandas DataFrame containing the matches (Mentee ID, Mentee Name, Assigned Mentor ID, Assigned Mentor Name, Similarity Score).
    *   Saves this DataFrame to an Excel file (`mentor_mentee_matches.xlsx` by default in the script's execution directory).

## Key Components

*   **Configuration Constants**:
    *   `MENTEE_FILE_PATH`, `MENTOR_FILE_PATH`: Paths to the input Excel files. **Remember to update these if your files are named differently or located elsewhere.**
    *   `OUTPUT_FILE_PATH`: Path where the results Excel file will be saved.
    *   `MENTEE_TARGET_SEMANTIC_COLS`, `MENTOR_TARGET_SEMANTIC_COLS`: Lists defining which *standardized* column names contribute to the semantic profile used for matching.
*   **Helper Functions**:
    *   `clean_text(text)`: Performs basic text cleaning (lowercase, remove extra whitespace).
    *   `combine_text_fields(row, columns)`: Concatenates text from specified columns in a DataFrame row into a single string, separated by ". ".
    *   `standardize_col_names(df)`: Implements the two-step column name standardization process (initial automatic cleaning + manual renaming based on a predefined map). **Crucial for ensuring the script finds the correct data.** Includes debugging prints to show columns before and after renaming.
*   **Main Script Logic**: Executes the workflow described above, including data loading, preprocessing, embedding, similarity calculation, matching, and saving results. Includes error handling for file loading and checks for required columns.

## How it Comes Together

The script aims to automate and improve the mentorship matching process by going beyond simple keyword matching.

1.  **Standardization is Key**: The `standardize_col_names` function is critical. It ensures that regardless of minor variations in the column headers of the input Excel files (e.g., "Why are you interested..." vs. "Reason for participating"), the script can reliably find and use the intended data by mapping them to consistent internal names (like `reason_for_participating`). If the script fails with missing columns, the first step is usually to check the `rename_map` within this function and compare it against the actual column headers in your input files (using the debug output helps here).
2.  **Semantic Profiles**: Instead of matching based on isolated fields, the script creates a holistic `semantic_profile` by combining text from multiple relevant columns. This allows the Sentence Transformer model to get a richer understanding of the individual's goals, interests, and skills.
3.  **Embeddings & Similarity**: The core of the "smart" matching lies in converting these profiles into embeddings. The Sentence Transformer model understands that phrases like "leading project teams" and "managing development initiatives" are semantically similar, even though they use different words. The cosine similarity calculation then quantifies this relatedness.
4.  **Constraints**: The matching isn't purely based on similarity. It incorporates practical constraints:
    *   **Manager Conflict**: Prevents direct manager-mentee pairings.
    *   **Mentor Capacity**: Ensures mentors aren't assigned more mentees than they indicated they can handle.
5.  **Output**: Provides a clear list of suggested pairings with the similarity score, allowing organizers to review the algorithm's suggestions.

## Usage

1.  **Prepare Data**: Ensure you have mentee and mentor data in Excel files (`.xlsx`). Verify the column headers.
2.  **Update Paths**: Modify `MENTEE_FILE_PATH` and `MENTOR_FILE_PATH` constants in the script if your filenames or locations differ from the defaults.
3.  **Customize Columns (If Necessary)**:
    *   If your input files have significantly different column headers than expected by the `rename_map` in `standardize_col_names`, update the map accordingly. Pay attention to the debug output showing column names before and after renaming.
    *   If you want to include different or additional fields in the semantic matching, update the `MENTEE_TARGET_SEMANTIC_COLS` and `MENTOR_TARGET_SEMANTIC_COLS` lists with the correct *standardized* column names.
4.  **Install Dependencies**: Make sure you have the required Python libraries installed:
    ```bash
    pip install pandas openpyxl sentence-transformers scikit-learn numpy regex
    ```
5.  **Run Script**: Execute the Python script from your terminal:
    ```bash
    python code/mentmatch.py
    ```
6.  **Review Results**: Check the generated `mentor_mentee_matches.xlsx` file for the pairings.

This detailed process allows for a nuanced and data-driven approach to mentorship matching, aiming for more compatible and successful pairings.
