<!DOCTYPE html>

<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="style.css">

  <title>Conditional Random Fields</title>
  <style>
    link rel="stylesheet" href="style.css">
    #about img {
      display: block;
      margin: 0 auto;
    }

  </style>
</head>
<body>
  <header>
    <h1>Welcome to Conditional Random Fields in Machine Learning</h1>
  </header>

  <nav>
    <!-- Add navigation links here -->
  </nav>

  <section id="about">
    <h2>About Conditional Random Fields</h2>
    <p>

Conditional random fields (CRFs) are a class of statistical modeling methods often applied in pattern recognition and machine learning and used for structured prediction. Whereas a classifier predicts a label for a single sample without considering "neighbouring" samples, a CRF can take context into account. To do so, the predictions are modelled as a graphical model, which represents the presence of dependencies between the predictions. What kind of graph is used depends on the application.</p>

<p>
            examples where CRFs are used are: labeling or parsing of sequential data for natural language processing or biological sequences:
<ol> 
<li>	Part-of-speech tagging and shallow pharsing. </li>
<li>	Named Entity Recognition. </li>
<li>	Gene Finding and Peptide critical functional region finding. </li>
<li>	Object Recognition. </li>
<li>	Image Segmentation in Computer vision. </li>
</ol>
    </p>
  </section>

  <section id="applications">
    <h2>Applications of Conditional Random Fields:</h2>
    <ul>
      <li>Named Entity Recognition (NER): CRFs are often used for NER tasks, where the goal is to identify and classify entities such as names of people, organizations, locations, etc. in a given text sequence.</li>
      <li>Part-of-Speech Tagging (POS): CRFs can be applied to POS tagging, which involves assigning grammatical categories (such as noun, verb, adjective) to each word in a sentence.</li>
      <li>Speech Recognition: CRFs can be used to model the dependencies between phonemes or words in speech recognition tasks, helping improve accuracy by considering context.</li>
      <li>Segmentation: CRFs can be used for tasks like image segmentation or video scene segmentation, where the goal is to label different parts of an image or video with appropriate categories.</li>
      <li>Gene Prediction in Bioinformatics: In computational biology, CRFs can be used to predict gene locations in DNA sequences by modeling the dependencies between nucleotides.</li>
      <li>Handwriting Recognition: CRFs can be applied to handwriting recognition to model the dependencies between the different strokes or components of the characters.</li>
      <li>Natural Language Processing (NLP): Beyond NER and POS tagging, CRFs have been used in various NLP tasks, including syntactic and semantic parsing, semantic role labeling, and more.</li>
      <li>Machine Translation: CRFs can assist in tasks related to the machine translation by modeling the dependencies between words in the source and target languages.</li>
      <li>Information Extraction: CRFs can help to extract structured information from unstructured text, such as extracting relationships between entities or the events.</li>
      <li>Video Analysis: In video analysis tasks like action recognition, CRFs can be used to model the temporal dependencies between actions in a sequence of video frames.</li>
      <li>Hand Gesture Recognition: CRFs can be used in gesture recognition systems to model the spatial and temporal dependencies between different parts of a gesture.</li>
      <li>Document Layout Analysis: CRFs can assist in tasks like document layout analysis, where the goal is to identify and categorize different elements within a document, such as headers, paragraphs, tables, etc.</li>
    </ul>
    <div style="text-align: center;"><!-- Centering the content -->

    </div>

  </section>

  <section id="papers">
    <h2>Classical Papers</h2>
    <ul>
      <li><a href="https://www.researchgate.net/publication/228057950_Conditional_Random_Fields_Probabilistic_Models_for_Segmenting_and_Labeling_Sequence_Data">Paper 1 :Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data</a></li>
      <li><a href="https://www.jmlr.org/papers/volume8/sutton07a/sutton07a.pdf">Paper 2 : Dynamic Conditional Random Fields</a></li>
      <li><a href="https://www.inference.org.uk/hmw26/crf/">Paper 3: Conditional Random Fields</a></li>
      <!-- Add more paper references as needed -->
    </ul>
  </section>

  <section id="Videos">
    <h2>Videos</h2>
    <ul>
      <li><a href="https://youtu.be/qQEQiPAxFxM">Video 1: Conditional Random Fields</a></li>
      <li><a href="https://youtu.be/rI3DQS0P2fk">Video 2: Conditional Random Fields:Data Science Concepts</a></li>
      <!-- Add more video references as needed -->
    </ul>
  <section id="code-section">
  <h2>Python Code Example: </h2>

  <pre>
    <code>
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Sample data
# Each sentence is represented as a list of dictionaries, where each dictionary has 'word' and 'label' keys.
# Labels can be 'B-PER' (beginning of a person entity), 'I-PER' (inside a person entity), 'O' (outside entity).
train_data = [
    [{'word': 'John', 'label': 'B-PER'}, {'word': 'Doe', 'label': 'I-PER'}, {'word': 'works', 'label': 'O'}],
    [{'word': 'Alice', 'label': 'B-PER'}, {'word': 'Smith', 'label': 'I-PER'}, {'word': 'is', 'label': 'O'}, {'word': 'an', 'label': 'O'}, {'word': 'engineer', 'label': 'O'}]
]

test_data = [
    [{'word': 'David', 'label': 'B-PER'}, {'word': 'Brown', 'label': 'I-PER'}, {'word': 'is', 'label': 'O'}, {'word': 'a', 'label': 'O'}, {'word': 'doctor', 'label': 'O'}]
]

# Feature extraction function
def word2features(sent, i):
    word = sent[i]['word']
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
    }
    if i > 0:
        features.update({
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        features.update({
            'word[:3]': word[:3],
            'word[:2]': word[:2],
        })
    else:
        features['EOS'] = True

    return features

# Convert data into features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [token['label'] for token in sent]

X_train = [sent2features(sent) for sent in train_data]
y_train = [sent2labels(sent) for sent in train_data]

X_test = [sent2features(sent) for sent in test_data]
y_test = [sent2labels(sent) for sent in test_data]

# Create and train CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Make predictions
y_pred = crf.predict(X_test)

# Evaluate the model
report = metrics.flat_classification_report(y_test, y_pred)
print(report)


    </code>
  </pre>
</section>

  <section id="AI-VIDEO">
    <h2>Embedded AI-Video</h2>
    <iframe width="560" Height="315" src="https://youtu.be/y_GvFpjQvTA" title="You Tube Video Player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; gyroscope; picture-in-picture" allowfullscreen></iframe>
  </section>
  
 <section id="AI-Video">
    <h2>Embedded AI-Video</h2>
    <iframe src="ttps://youtu.be/y_GvFpjQvTA" width="800" height="600" frameborder="0" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
  </section>
  

  <section id="presentation">
    <h2>Embedded Presentation</h2>
    <iframe src="https://docs.google.com/document/d/18wFNr9XqbxAwMj8l3H9YOwq1NSc6-4xy/edit?usp=sharing&ouid=101506226181491048107&rtpof=true&sd=true" width="800" height="600" frameborder="0" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
  </section>

  <footer>
    <p>&copy; 2023 Anirudha Mishra. All rights reserved.</p>
  </footer>
</body>
</html>
