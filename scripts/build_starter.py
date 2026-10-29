"""Deterministically build DRAFT Odia/English astronomy examples from saved NASA sources.

This authors short paraphrases, not page-extraction ground truth. Page numbers below
refer to sections in our corpus, not the original HTML. Odia requires fluent review.
"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URLS = {n: f"https://science.nasa.gov/{n}/facts/" for n in ("earth", "moon", "sun")}

PARAGRAPHS = [
    ("earth", 1, "Earth is the third planet from the Sun.", "ପୃଥିବୀ ସୂର୍ଯ୍ୟଠାରୁ ତୃତୀୟ ଗ୍ରହ।"),
    ("earth", 2, "Earth takes 365.25 days to complete one orbit around the Sun.", "ସୂର୍ଯ୍ୟ ଚାରିପାଖରେ ଥରେ ପରିକ୍ରମା କରିବା ପାଇଁ ପୃଥିବୀକୁ ୩୬୫.୨୫ ଦିନ ଲାଗେ।"),
    ("earth", 3, "The global ocean covers about 71 percent of Earth's surface.", "ପୃଥିବୀ ପୃଷ୍ଠର ପ୍ରାୟ ୭୧ ପ୍ରତିଶତ ଅଂଶ ସମୁଦ୍ରରେ ଆଚ୍ଛାଦିତ।"),
    ("earth", 4, "Earth completes one rotation in about 23.9 hours.", "ପୃଥିବୀ ପ୍ରାୟ ୨୩.୯ ଘଣ୍ଟାରେ ନିଜ ଅକ୍ଷ ଚାରିପାଖରେ ଥରେ ଘୂରେ।"),
    ("moon", 1, "The Moon is Earth's only natural satellite.", "ଚନ୍ଦ୍ର ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ।"),
    ("moon", 2, "The Moon is about 385,000 kilometers from Earth.", "ଚନ୍ଦ୍ର ପୃଥିବୀଠାରୁ ପ୍ରାୟ ୩୮୫,୦୦୦ କିଲୋମିଟର ଦୂରରେ ଅଛି।"),
    ("sun", 1, "The Sun is a star at the center of our solar system.", "ସୂର୍ଯ୍ୟ ଆମ ସୌରମଣ୍ଡଳର କେନ୍ଦ୍ରରେ ଥିବା ଏକ ନକ୍ଷତ୍ର।"),
    ("sun", 2, "The Sun's gravity holds our solar system together.", "ସୂର୍ଯ୍ୟର ମହାକର୍ଷଣ ଶକ୍ତି ଆମ ସୌରମଣ୍ଡଳକୁ ଏକତ୍ର ବାନ୍ଧି ରଖେ।"),
]
# source/page references are resolved against language-specific corpus documents.
SPECS = [
    ("Which planet from the Sun is Earth?", "ସୂର୍ଯ୍ୟଠାରୁ ପୃଥିବୀ କେତେ ନମ୍ବର ଗ୍ରହ?", [("earth",1)], "direct", "Surjyatharu Pruthibi kete number graha?"),
    ("How many days does Earth take to orbit the Sun?", "ସୂର୍ଯ୍ୟ ଚାରିପାଖରେ ପରିକ୍ରମା କରିବାକୁ ପୃଥିବୀକୁ କେତେ ଦିନ ଲାଗେ?", [("earth",2)], "direct", "Pruthibi Sun charipakhare orbit karibaku kete days lage?"),
    ("What percentage of Earth's surface does the ocean cover?", "ପୃଥିବୀ ପୃଷ୍ଠର କେତେ ପ୍ରତିଶତ ଅଂଶ ସମୁଦ୍ରରେ ଆଚ୍ଛାଦିତ?", [("earth",3)], "direct", "ପୃଥିବି surface ର କେତେ percent ସମୁଦ୍ର?"),
    ("What is Earth's only natural satellite?", "ପୃଥିବୀର ଏକମାତ୍ର ପ୍ରାକୃତିକ ଉପଗ୍ରହ କିଏ?", [("moon",1)], "direct", "Pruthibira ekamatra prakrutika upagraha kie?"),
    ("How far is the Moon from Earth in kilometers?", "ଚନ୍ଦ୍ର ପୃଥିବୀଠାରୁ କେତେ କିଲୋମିଟର ଦୂରରେ ଅଛି?", [("moon",2)], "direct", "ଚନ୍ଦ୍ର ପୃଥିବୀଠାରୁ କେତେ kilometer ଦୂରରେ?"),
    ("What kind of object is the Sun and where is it in our solar system?", "ସୂର୍ଯ୍ୟ କେଉଁ ପ୍ରକାରର ଜ୍ୟୋତିଷ୍କ ଏବଂ ସୌରମଣ୍ଡଳର କେଉଁଠାରେ ଅଛି?", [("sun",1)], "direct", "ସୁର୍ଯ୍ୟ କଣ ଏବଂ solar system ର କେଉଁଠି ଅଛି?"),
    ("How long does Earth take to rotate once and orbit the Sun once?", "ପୃଥିବୀର ଥରେ ଘୂର୍ଣ୍ଣନ ଏବଂ ସୂର୍ଯ୍ୟ ଚାରିପାଖରେ ଥରେ ପରିକ୍ରମା ପାଇଁ କେତେ ସମୟ ଲାଗେ?", [("earth",4),("earth",2)], "multi-page", "ପୃଥିବୀର 1 ଥର rotation ଓ ୧ ଥର revolution ପାଇଁ କେତେ ସମୟ ଲାଗେ?"),
    ("What kind of object is the Sun, and which planet from it is Earth?", "ସୂର୍ଯ୍ୟ କେଉଁ ପ୍ରକାରର ଜ୍ୟୋତିଷ୍କ ଏବଂ ପୃଥିବୀ ତାହାଠାରୁ କେତେ ନମ୍ବର ଗ୍ରହ?", [("sun",1),("earth",1)], "cross-document", "Surjya kana ebam Pruthibi kete number graha?"),
    ("What is the unknown opening date of a future museum on the Moon?", "ଚନ୍ଦ୍ରରେ ଭବିଷ୍ୟତର ଏକ ସଂଗ୍ରହାଳୟ ଖୋଲିବାର ଅଜଣା ତାରିଖ କଣ?", [], "unanswerable", "Chandra re museum kholibara unknown date kana?"),
    ("What is the unknown exact date of the next discovery of life on another planet?", "ଅନ୍ୟ ଗ୍ରହରେ ଜୀବନ ଆବିଷ୍କାରର ଆଗାମୀ ଅଜଣା ତାରିଖ କଣ?", [], "unanswerable", "Anya grahare life discovery ra unknown date kana?"),
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build():
    chunks = []
    for topic, page, en, odia in PARAGRAPHS:
        for lang, text in (("en",en),("or",odia)):
            chunks.append({"id":f"{topic}-{lang}-{page}","text":text,"source":f"{topic}-{lang}","page":page,"language":lang,"source_url":URLS[topic]})
    write_jsonl(DATA / "starter_corpus.jsonl", chunks)
    lookup = {(t,p): (en,odia) for t,p,en,odia in PARAGRAPHS}
    cases = []
    for i,(en,odia,pages,kind,variant) in enumerate(SPECS,1):
        tracks = [("en","en","en",en,"Latn","core"),("or","or","or",odia,"Orya","core"),
                  ("or","en","or",odia,"Orya","core"),("en","or","en",en,"Latn","core"),
                  ("or","mixed","or",odia,"Orya","mixed"),("or","mixed","or",variant,"mixed" if any("\u0b00"<=c<="\u0b7f" for c in variant) else "Latn","robustness")]
        for qlang,clang,alang,query,script,track in tracks:
            languages=["or","en"] if clang=="mixed" else [clang]
            facts=[lookup[p][0 if alang=="en" else 1] for p in pages]
            if not facts:
                facts=["The provided documents don't answer this." if alang=="en" else "ଦିଆଯାଇଥିବା ଦଲିଲରେ ଏହାର ଉତ୍ତର ନାହିଁ।"]
            labels=[{"source":f"{t}-{lang}","page":p,"grade":2} for t,p in pages for lang in languages]
            cases.append({"id":f"case-{i:02}-{qlang}-{clang}-{track}","semantic_case_id":f"astronomy-{i:02}","query":query,"query_language":qlang,"query_script":script,"answer_language":alang,"corpus_languages":languages,"question_type":kind if track!="robustness" else "robustness","review_status":"draft","expected_answer":" ".join(facts),"key_facts":facts,"relevant_pages":labels,"unanswerable":not pages})
    write_jsonl(DATA / "starter_cases.jsonl", cases)
    first=next(c for c in cases if c["id"]=="case-01-or-or-core")
    fixed_context=[c for c in chunks if c["id"]=="earth-or-1"]
    calibration=[]
    for name,answer in [("grounded",fixed_context[0]["text"]),("invented","ପୃଥିବୀ ପନିରରେ ତିଆରି ହୋଇଛି।"),("incorrect-refusal","ଦିଆଯାଇଥିବା ଦଲିଲରେ ଏହାର ଉତ୍ତର ନାହିଁ।")]:
        case=dict(first);case["id"]=f"calibration-{name}"
        calibration.append({"case":case,"context":fixed_context,"answer":answer,"label_review_status":"draft","expected_judgment":None})
    write_jsonl(DATA/"calibration_draft.jsonl",calibration)
    manifest={"review_status":"draft","description":"Authored English paraphrases and unreviewed Odia translations of saved NASA facts. Page means authored corpus section. Not extracted PDF page numbers.","semantic_cases":10,"query_instances":60,"sources":[]}
    for topic,url in URLS.items():
        p=DATA/"sources"/f"{topic}.html"
        if not p.exists():
            raise FileNotFoundError(f"Missing source snapshot {p}")
        manifest["sources"].append({"url":url,"snapshot":f"sources/{topic}.html","sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (DATA/"starter_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")


def reference():
    source=ROOT.parent/"RAG-ReRanker-Comparison"/"eval"/"eval_dataset.py"
    if not source.exists():
        return
    rows=[]
    for i,c in enumerate(runpy.run_path(str(source))["EVAL_DATASET"],1):
        rows.append({"id":f"reference-{i:02}","semantic_case_id":f"reference-{i:02}","query":c["query"],"query_language":"en","query_script":"Latn","answer_language":"en","corpus_languages":["en"],"question_type":c["type"],"review_status":"draft","expected_answer":c["expected_answer"],"key_facts":c["key_facts"],"relevant_pages":[{"source":s,"page":p,"grade":g} for s,p,g in c["relevant_pages"]],"unanswerable":not c["relevant_pages"]})
    write_jsonl(DATA/"reference_cases.jsonl",rows)


if __name__=="__main__":
    build()
    reference()
