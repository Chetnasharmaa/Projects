import nltk 
from nltk.chat.util import Chat,reflections
reflections = {

"i am" : "you are",

"i was" : "you were",

"i" : "you",

"i'm" : "you are",

"i'd" : "you would",

"i've" : "you have",

"i'll" : "you will",

"my" : "your",

"you are" : "I am",

"you were" : "I was",

"you've" : "I have",

"you'll" : "I will",

"your" : "my",

"yours" : "mine",

"you" : "me",

"me" : "you"

}
pairs = [

[
# r --> to indentify if its input from user or not
#.* the users enters and %1--> same as user enteres
r"my name is (.*)",

["Hello %1, How are you today ?",]

],

[

r"hi|hey|hello",

["Hello", "Hey there",]

],

[

r"what is your name ?",

["I am a bot created by Codingal Edu. pvt. Lim. you can call me Jarvis!",]

],

[

r"how are you ?",

["I'm doing goodnHow about You ?",]

],

[

r"sorry (.*)",

["Its alright","Its OK, never mind",]

],

[

r"I am fine",

["Great to hear that, How can I help you?",]

],

[

r"i'm (.*) doing good",

["Nice to hear that","How can I help you?:)",]

],

[

r"(.*) age?",

["I'm a computer program dudenSeriously you are asking me this?",]

],

[

r"what (.*) want ?",

["Make me an offer I can't refuse",]

],

[

r"(.*) created ?",

["Shravan created me using Python's NLTK library ","top secret ;)",]

],

[

r"(.*) (location|city) ?",

['Bangalore, Karnataka',]

],

[

r"how is weather in (.*)?",

["Weather in %1 is awesome like always","Too hot man here in %1","Too cold man here in %1","Never even heard about %1"]

],

[

r"i work in (.*)?",

["%1 is an Amazing company, I have heard about it. But they are in huge loss these days.",]

],

[

r"(.*)raining in (.*)",

["No rain since last week here in %2","Damn its raining too much here in %2"]

],

[

r"how (.*) health(.*)",

["I'm a computer program, so I'm always healthy ",]

],

[

r"(.*) (sports|game) ?",

["I'm a very big fan of Football and Cricket",]

],

[

r"who (.*) sportsperson ?",

["Messy","Ronaldo","Roony", "Virat", "M.S. Dhoni", "Rohit"]

],

[

r"who (.*) (moviestar|actor)?",

["Benedict Cumberbatch"]

],

[

r"i am looking for online guides and courses to learn data science, can you suggest?",

["Jarvis_Tech has many great articles with each step explanation along with code, you can explore"]

],

[

r"quit",

["BBye take care. See you soon :) ","It was nice talking to you. See you soon :)"]

],

]
def chat():
    print("Hi! I am your chatbot!!")
    chat=Chat(pairs,reflections)
    chat.converse()
    #simplify into readable format from the module. We mostly work in Large Language Modules.
if __name__=="__main__":
    #constructor when object are created
    chat()


