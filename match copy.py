import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import extract


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_text, job_description_text):
    # Tokenization and normalization
    vectorizer = TfidfVectorizer()
    corpus = [resume_text, job_description_text]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Return the similarity score
    return similarity_matrix[0][1]




def main():

    docx_path = r'C:\Users\shrut\OneDrive\Desktop\tejoma\Vaishnavi.Tammali.docx' 
    pdf_path = r'C:\Users\shrut\OneDrive\Desktop\tejoma\Vaishnavi.Tammali.pdf'


    # resume_text = extract.read_word_document(docx_path)
    resume_text = extract.extract_text_from_pdf(pdf_path)
    job_description_text = """
    Desired Experience Range: 4 to 8 Years

    Job Location: Chennai, Hyderabad, Bangalore, Mumbai, Pune



    Required Skill Set: R, Python, and database query languages like SQL, Hive, Pig is desirable. Familiarity with Scala, Java, or C++, Machine learning , Data Wrangling



    Must-Have: Python, Machine learning, data Wrangline
    Good-to-Have: Hadoop, Spark


    Expectations from the Role:

    1. Programming Skills - knowledge of statistical programming languages like R, Python, and database query languages like SQL, Hive, Pig is desirable. Familiarity with Scala, Java, or C++ is an added advantage.
    2. Statistics - Good applied statistical skills, including knowledge of statistical tests, distributions, regression, maximum likelihood estimators, etc. Proficiency in statistics is essential for data-driven companies.
    3. Machine Learning - good knowledge of machine learning methods like k-Nearest Neighbors, Naive Bayes, SVM, Decision Forests.
    4. Hands-on experience with data science tools
    5. Knowledge in Spark is added advantage
    6. Knowledge in Hadoop is added advantage
    7. Proven Experience as Data Analyst or Data Scientist
    8. Experience in large scale enterprise application implementation
    9. Creative Individual with a track record of working on and implementing innovative tech based solutions
    10. Exceptional intelligence and problem solving skills
    """

    similarity_score = calculate_similarity(resume_text, job_description_text)
    similarity_percentage = similarity_score*100
    print(f"Similarity percentage : {similarity_percentage}")

if __name__ == "__main__":
    main()
