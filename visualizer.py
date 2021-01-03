import en_core_web_md
from spacy import displacy
nlp = en_core_web_md.load()

doc = nlp("A Texas professor who helped mount President Nixon 's unsuccessful attempt to resist a key Watergate subpoena is being considered for the post of U.S. solicitor .")

displacy.serve(doc)