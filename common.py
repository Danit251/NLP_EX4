from collections import namedtuple

RELATION = "Work_For"
PERSON = "PER"
ORG = "ORG"
TEXT = "text"
TYPE = "type"
SPAN = "span"


Relation = namedtuple('Relation', ['person', 'org', 'sentence'])
def is_the_same(s1, s2):
    return s1 == s2 or s1 in s2 or s2 in s1
