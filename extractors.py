from flair.data import Sentence
from flair.models import SequenceTagger
# check = 'Also being considered are Judge Ralph K. Winter of the 2nd U.S. Circuit Court of Appeals in New York City and Judge Kenneth Starr of the U.S. Circuit Court of Appeals for the District of Columbia , said the source , who spoke on condition of anonymity .'
# check = "Tony Denny , executive director of the state Republican Party , said he didn 't know of any mix-up , but that Mary Campbell of Clover is free to keep her tickets ."
check = "Larry Jinks has been named publisher of the San Jose Mercury News , succeeding William A. Ott , who will become the newspaper 's chairman , the Knight-Ridder , Inc. announced Thursday ."
# make a sentence
sentence = Sentence(check)

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)
for entity in sentence.get_spans('ner'):
    print(entity)
