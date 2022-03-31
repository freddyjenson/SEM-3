import nltk

text = "India is my country. " \
       "All indians are my brothers and sisters." \
       "I love my country." \
       "India is a diverse country." \
       "It is rich in its cultures"

sent = nltk.sent_tokenize(text)

print(sent)

for i in sent:
    token = nltk.word_tokenize(i)

    print(token)

    tags = nltk.pos_tag(token)

    print(tags)