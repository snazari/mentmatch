import pandas as pd
import os

# Define data for mentees
mentee_data = {
    'ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Manager': ['Frank', 'Grace', 'Frank', 'Heidi', 'Ivan'],
    'Competencies you would like to work on with a mentor': [
        'Improve Python programming skills and data analysis techniques',
        'Develop project management and leadership abilities',
        'Enhance public speaking and presentation confidence',
        'Learn about cloud computing and DevOps practices',
        'Gain experience in machine learning and AI applications'
    ],
    'Why are you interested in participating in the mentoring program': [
        'Career growth and skill development',
        'Networking and learning from experienced professionals',
        'Guidance on navigating corporate structure',
        'Exploring new technical domains',
        'Preparing for a future leadership role'
    ],
    'What is your favorite activity/hobby': ['Reading sci-fi novels', 'Playing chess', 'Hiking and camping', 'Building custom PCs', 'Photography'],
    'What is your favorite movie genre': ['Science Fiction', 'Thriller', 'Adventure', 'Documentary', 'Comedy'],
    'What is your favorite book genre': ['Science Fiction', 'Mystery', 'Fantasy', 'Non-fiction', 'Humor'],
    'What is a fun fact about you': ['Can speak Klingon', 'Completed a marathon', 'Visited 5 continents', 'Has a pet snake', 'Collects vintage stamps']
}

# Define data for mentors
mentor_data = {
    'ID': [201, 202, 203, 204],
    'Name': ['Frank', 'Grace', 'Heidi', 'Ivan'],
    # Using varied inputs for capacity cleaning test
    'How many mentees are you able to mentor throughout the program': ['One', 2, 'one', '3'],
    'As a mentor... please select the competencies you feel you could teach...': [
        'Expert Python development, data science fundamentals, and team leadership',
        'Agile project management, stakeholder communication, and risk assessment',
        'Effective presentation skills, public speaking coaching, and storytelling',
        'Cloud architecture (AWS/Azure), CI/CD pipelines, and infrastructure as code'
    ],
    'Why are you interested in participating in the mentoring program': [
        'Share knowledge and give back to the community',
        'Develop coaching skills and learn from mentees',
        'Help shape future leaders',
        'Expand professional network'
    ],
    'What is your favorite activity/hobby': ['Playing guitar', 'Cooking gourmet meals', 'Running marathons', 'Volunteering at animal shelter'],
    'What is your favorite movie genre': ['Drama', 'Comedy', 'Action', 'Documentary'],
    'What is your favorite book genre': ['Biographies', 'Cookbooks', 'Thriller', 'History'],
    'What is a fun fact about you': ['Was a world champion Catan player', 'Lived abroad for 5 years', 'Climbed Mt. Kilimanjaro', 'Designs board games']
}

# Create DataFrames
df_mentees = pd.DataFrame(mentee_data)
df_mentors = pd.DataFrame(mentor_data)

# Ensure the data directory exists
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Define output paths
mentee_output_path = os.path.join(data_dir, 'mentee_test.xlsx')
mentor_output_path = os.path.join(data_dir, 'mentor_test.xlsx')

# Save to Excel files
try:
    df_mentees.to_excel(mentee_output_path, index=False)
    df_mentors.to_excel(mentor_output_path, index=False)
    print(f"Successfully created '{mentee_output_path}'")
    print(f"Successfully created '{mentor_output_path}'")
except Exception as e:
    print(f"Error creating test data files: {e}")
