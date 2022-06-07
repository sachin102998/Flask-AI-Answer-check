# #!/usr/local/bin/python/python.exe

# print "Content-type: text/html\n\n";

# import cgi
# form=cgi.FieldStorage()

# name=form.getValue("name")
# print(name)
import math

from flask import Flask, render_template,request
app = Flask(__name__)
# import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download() 
from final import ngram,summary,eval_marks
orig='In object-oriented programming, inheritance is a way to form new classes (instances of which are called objects) using classes that have already been defined. The inheritance concept was invented in 1967 for Simula. The new classes, known as derived classes, take over (or inherit) attributes and behavior of the pre-existing classes, which are referred to as base classes (or ancestor classes). It is intended to help reuse existing code with little or no modification. Inheritance provides the support for representation by categorization in computer languages. Categorization is a powerful mechanism number of information processing, crucial to human learning by means of generalization (what is known about specific entities is applied to a wider group given a belongs relation can be established) and cognitive economy (less information needs to be stored about each specific entity, only its particularities). Inheritance is also sometimes called generalization, because the is-a relationships represent a hierarchy between classes of objects. For instance, a "fruit" is a generalization of "apple", "orange", "mango" and many others. One can consider fruit to be an abstraction of apple, orange, etc. Conversely, since apples are fruit (i.e., an apple is-a fruit), apples may naturally inherit all the properties common to all fruit, such as being a fleshy container for the seed of a plant. An advantage of inheritance is that modules with sufficiently similar interfaces can share a lot of code, reducing the complexity of the program. Inheritance therefore has another view, a dual, called polymorphism, which describes many pieces of code being controlled by shared control code. Inheritance is typically accomplished either by overriding (replacing) one or more methods exposed by ancestor, or by adding new methods to those exposed by an ancestor. Complex inheritance, or inheritance used within a design that is not sufficiently mature, may lead to the Yo-yo problem.'



@app.route("/")
def index():
  return render_template("index.html")
  
@app.route("/compute",methods=['POST'])
def compute():
    name = request.form['name']
    email = request.form['email']
    roll=request.form['rollno']
    inst_1=request.form['answer']
    score=math.floor(eval_marks(orig,inst_1))
    return render_template("marks.html",name=name,value=score,roll=roll)
    #return 'You got %s have fun learning python <br/> <a href="/">Back Home</a>' %(score*100)
 

if __name__ == "__main__":
  app.run(debug=True)