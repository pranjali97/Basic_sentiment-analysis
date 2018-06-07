import speech_recognition as sr
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

r = sr.Recognizer()
m = sr.Microphone()

def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)', 'amazing', 'marvelous', 'cool', 'great', 'perfect']
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 
train_set = negative_features + positive_features + neutral_features
 
classifier = NaiveBayesClassifier.train(train_set) 
 

neg = 0
pos = 0

try:
    print("A moment of silence, please...")
    with m as source:
        r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    while True:
        print("Say something!")
        with m as source:
            audio = r.listen(source)
        print("Got it! Trying to recognize...")
        try:
            # speech to text
            value = r.recognize_google(audio)
            print("You said ",value)
            words = value.split(' ')
            for word in words:
                classResult = classifier.classify( word_feats(word))
                if classResult == 'neg':
                    neg = neg + 1
                if classResult == 'pos':
                    pos = pos + 1 
            print('Positive: ' + str(float(pos)/len(words)))          
            print('Negative: ' + str(float(neg)/len(words))) 

        except sr.UnknownValueError:
            print("Oops! Didn't catch that")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
	
except Exception as e:
	raise