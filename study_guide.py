#find the words that are likely to mater 
import fasttext as ft
import nltk
import spacy

#does word equal word from before
data = "C:/Users/alexk/Downloads/SYLLABUS.md"

def break_sentences(txt: str) -> list[str]:
    """Break text into a list of sentences"""
    return nltk.tokenize.sent_tokenize(txt)

"""
Term Frequency and Inverse Document Frequncy are ideas we can use to build 
this study guide quickly.

After converting words to vectors, vectors that are similar to 'data science' a.e same
vector area thingy are words we prioritize
"""
class NounParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def phrases(self, txt: str) -> list[str]:
        """Convert text into noun phrases"""
        doc = self.nlp(txt.lower())
        words = []
        for phrase in doc.noun_chunks:
            word = phrase.text
            word = word.strip()
            if not (word == "" or word.isspace()):
                words.append(word.replace("\n", ""))
        return words

data_bs = break_sentences(data)
np = NounParser()
data_np = np.get_phrases(data_bs)

ft_model = ft.load_model("data/cc.en.50.bin")
data_ft = ft_model.get_sentence_vector(data_bs)
print(data_ft)
