import PyPDF2
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Open the PDF file and read the contents
filename = input("Type file name: ")
pdf_file = open(str(filename), 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Preprocess the text using NLTK
sentences = nltk.sent_tokenize(text)
words = [nltk.word_tokenize(sentence) for sentence in sentences]
words = [[word.lower() for word in sentence if word.isalpha()] for sentence in words]

# Apply unsupervised learning (e.g. topic modeling) to understand the text
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform([' '.join(sentence) for sentence in words])
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

# Answer questions related to the text
while True:
    question = input("What do you want to know? (Press Q to quit)\n")
    if question.lower() == 'q':
        break
    question_words = nltk.word_tokenize(question.lower())
    if 'main topic' in question:
        topic_words = [vectorizer.get_feature_names()[i] for i in lda.components_[0].argsort()[-10:]]
        print(f'The main topic of the text is: {" ".join(topic_words)}')
    elif 'pages' in question:
        print(f'The text has {len(pdf_reader.pages)} pages.')
    else:
        # Find the sentence in the text that is most similar to the question
        similarities = []
        for sentence in sentences:
            sentence_words = nltk.word_tokenize(sentence.lower())
            similarity = len(set(sentence_words).intersection(question_words)) / len(set(sentence_words).union(question_words))
            similarities.append(similarity)
        most_similar_sentence = sentences[similarities.index(max(similarities))]
        
        # Print the answer (the next sentence after the most similar sentence)
        index = sentences.index(most_similar_sentence)
        if index < len(sentences)-1:
            answer = sentences[index+1]
        else:
            answer = "Sorry, I don't know the answer."
        print(answer)
