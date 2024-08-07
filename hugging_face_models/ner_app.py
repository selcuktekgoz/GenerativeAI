import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
model = AutoModelForTokenClassification.from_pretrained(
    "savasy/bert-base-turkish-ner-cased"
)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def ner(text):
    output = nlp(text)
    return {"text": text, "entities": output}


examples = [
    "Bugün Niğde'de hava çok güneşli.",
    "İzmit te Selcuk ve annesi var.",
    "Bu yaz Antalya Side de Sunrise Beach Side Otele tatile gidelim mi?",
]

demo = gr.Interface(
    ner,
    gr.Textbox(placeholder="Enter sentence here..."),
    gr.HighlightedText(),
    examples=examples,
)
demo.launch(share=False)
