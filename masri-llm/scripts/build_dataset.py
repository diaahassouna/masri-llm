#!/usr/bin/env python3
"""
build_dataset.py — Turns Diaa Hassouna's Masri source-of-truth files
(alphabet.json, tier2-rules.json, masri_tier2_system_prompt.md,
masri_tier2_eval_set.json) into Hugging Face-ready SFT datasets.

Outputs (into ../data/):
  train.jsonl        — SFT training examples (chat format: system/user/assistant)
  eval_held_out.jsonl — eval-set-derived examples, kept OUT of train.jsonl on purpose
                         (this is masri_tier2_eval_set.json — use it only for scoring,
                         never for training, or your eval numbers become meaningless)
  dataset_stats.json — counts per category, for sanity-checking coverage

Run:
  python3 build_dataset.py
"""
import json
import random
from pathlib import Path

random.seed(42)

SRC = Path(__file__).parent.parent / "source"
OUT = Path(__file__).parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

with open(SRC / "alphabet.json", encoding="utf-8") as f:
    alphabet = json.load(f)
with open(SRC / "tier2-rules.json", encoding="utf-8") as f:
    rules = json.load(f)
with open(SRC / "masri_tier2_system_prompt.md", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()
with open(SRC / "masri_tier2_eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

examples = []  # list of dicts: {"messages": [...], "category": str, "source": str}


def add(user, assistant, category, source):
    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "category": category,
            "source": source,
        }
    )


# ---------------------------------------------------------------------------
# 1. Alphabet grounding — letter <-> Arabic <-> IPA <-> name, both directions.
#    This teaches the base model the actual symbol inventory before it ever
#    has to use the symbols in running text.
# ---------------------------------------------------------------------------
letter_q_templates_ar_to_masri = [
    "إيه الحرف بتاع {ar} في الأبجدية المصرية؟",
    "اكتبلي حرف {ar} بالأبجدية المصرية.",
]
letter_q_templates_masri_to_ar = [
    "الحرف {up}/{lo} في الأبجدية المصرية بيمثل إيه في العربي؟",
]

for letter in alphabet["alphabet"]:
    up, lo, ar, name, ipa = (
        letter["letter_upper"],
        letter["letter_lower"],
        letter["arabic"],
        letter["name"],
        letter["ipa"],
    )
    notes_ar = letter.get("notes_ar", "")
    q1 = random.choice(letter_q_templates_ar_to_masri).format(ar=ar)
    a1 = f"حرف {ar} في الأبجدية المصرية (الفئة الثانية) بيتكتب {up}/{lo} ({name}). {notes_ar}".strip()
    add(q1, a1, "alphabet", "alphabet.json")

    q2 = random.choice(letter_q_templates_masri_to_ar).format(up=up, lo=lo)
    a2 = f"{up}/{lo} ({name}, {ipa}) بيمثل حرف {ar} في العربي. {notes_ar}".strip()
    add(q2, a2, "alphabet", "alphabet.json")

for b in alphabet["borrowed_orthography"]:
    q = f"إمتى بستخدم حرف {b['letter_upper']}/{b['letter_lower']} في المصرية؟"
    a = f"{b['letter_upper']}/{b['letter_lower']} ({b['name']}) {b['notes_ar']}."
    add(q, a, "alphabet_borrowed", "alphabet.json")

# ---------------------------------------------------------------------------
# 2. Rule explanations — ask the model to state a rule, in Masri Tier 2 itself,
#    so the model practices *generating* Masri prose, not just converting into it.
# ---------------------------------------------------------------------------
RULE_EXPLANATIONS_MASRI = {
    "gemination": "El Ϩorouf el metⲴaddeda betetketeb metⲴadda fel kalema, ϨaⲴalāla ⲴaN maB6 el shadda maktouba weLLa la'. Zayy 'geddan' aw 'Ϩobb'.",
    "definite_article": "'El' betetketeb daayman keda, kelma leϨaaha, meϣ metdammega maⲴ elli baⲴdaha — meϣ 'elnaharda' walla 'el-naharda', laakenn 'el naharda'.",
    "glottal_stop": "Ɐ betetketeb bass fe nosṣ el kelma walla fe axerha, mesh fe awwelha. El hamza fe awwel el kelma dayman saakta we metmaktobaash.",
    "ayin": "Ⲵ (el Ⲵayn el Ⱳobṭeya el adeema) howa el Ϩarf elli beyeⲴber Ⲵan ص ع fel Ⲵarabi, fel Tier 2. Da axtar Ϩarf, laazem yetsahheb feeh koll marra.",
    "q_hamza_merger": "El Ⲵaammeya el Ɐaahereyya betⲴaamel el ق zayy el hamza — betwaddeeh Ɐ. Bass el q betfḍal lel formeyya aw el fosⲴa bass.",
}

add(
    "اشرحلي في المصرية إيه قاعدة الـgemination.",
    RULE_EXPLANATIONS_MASRI["gemination"],
    "rule_explanation",
    "tier2-rules.json + system prompt",
)
add(
    "قوللي بالمصري إمتى بنستخدم Ɐ.",
    RULE_EXPLANATIONS_MASRI["glottal_stop"],
    "rule_explanation",
    "tier2-rules.json",
)
add(
    "إيه أهم قاعدة في كتابة حرف العين بالمصرية؟ رد بالمصري.",
    RULE_EXPLANATIONS_MASRI["ayin"],
    "rule_explanation",
    "tier2-rules.json ayin_rule",
)

# ---------------------------------------------------------------------------
# 3. Ayin examples (both files) — direct conversion drills.
# ---------------------------------------------------------------------------
for ex in alphabet["ayin_rule"]["examples"]:
    add(
        f"حوّل الكلمة دي للمصرية (Tier 2): {ex['arabic']}",
        ex["tier2"],
        "ayin",
        "alphabet.json ayin_rule",
    )

for ex in rules["ayin_rule"]["examples"]:
    add(
        f"حوّل الكلمة دي للمصرية (Tier 2): {ex['arabic']} (معناها: {ex['meaning']})",
        ex["tier2"],
        "ayin",
        "tier2-rules.json ayin_rule",
    )

# ---------------------------------------------------------------------------
# 4. Spelling-rule worked examples (rule 1-6 blocks in tier2-rules.json)
# ---------------------------------------------------------------------------
for rule in rules["spelling_rules"]:
    for ex in rule.get("examples", []):
        if "→" not in ex:
            continue
        src, tgt = [p.strip() for p in ex.split("→", 1)]
        add(
            f"حوّل الكلمة/الجملة دي للمصرية (Tier 2) وطبّق قاعدة '{rule['name']}': {src}",
            tgt,
            f"rule_{rule['id']}_{rule['name'].lower().replace(' ', '_')}",
            "tier2-rules.json spelling_rules",
        )

# ---------------------------------------------------------------------------
# 5. Standardized high-frequency words — memorized closed list.
# ---------------------------------------------------------------------------
for w in rules["standardized_word_list"]:
    add(
        f"إزاي بتتكتب '{w['arabic']}' بالمصرية (Tier 2)؟",
        w["tier2"],
        "standardized_word",
        "tier2-rules.json standardized_word_list",
    )

# ---------------------------------------------------------------------------
# 6. Loanword policy examples (correct vs wrong spelling contrast)
# ---------------------------------------------------------------------------
for lw in rules["loanword_examples"]:
    correct = lw.get("tier1")
    wrong = lw.get("wrong")
    if not correct:
        continue
    if wrong:
        add(
            f"إيه الصح والغلط في كتابة '{lw['arabic']}' بالمصرية؟ ({lw.get('notes','')})",
            f"الصح: {correct}. الغلط الشائع: {wrong} — لازم نفرّق بين B/P عشان دول حروف مستقلة في المصرية.",
            "loanword_p_b_v_f",
            "tier2-rules.json loanword_examples",
        )
    else:
        add(
            f"إزاي بتتكتب '{lw['arabic']}' بالمصرية؟ ({lw.get('source','')})",
            correct,
            "loanword",
            "tier2-rules.json loanword_examples",
        )

# ---------------------------------------------------------------------------
# 7. Stress-test sentences — pure Masri monolingual text, used as
#    "continue/respond in Masri" conversational turns so the model learns
#    natural running prose, not just isolated word conversions.
# ---------------------------------------------------------------------------
conversation_openers = [
    "قوللي حاجة عن مصر بالمصري.",
    "اكتبلي جملة طويلة بالمصري Tier 2 تجرب فيها أكتر من قاعدة.",
    "عايز مثال جملة مصرية معقدة تستخدم فيها حروف قبطية ويونانية.",
    "احكيلي عن يومك بالمصري.",
    "قوللي رأيك في حاجة بالمصري.",
]
for i, sent in enumerate(rules["sample_texts"]["stress_test_sentences_tier2"]):
    opener = conversation_openers[i % len(conversation_openers)]
    add(opener, sent, "conversational_masri", "tier2-rules.json stress_test_sentences_tier2")

add(
    "احكيلي عن مصر والقهوة المصرية بالمصري.",
    rules["sample_texts"]["tier2_academic"],
    "conversational_masri",
    "tier2-rules.json sample_texts",
)
add(
    f"حوّل النص ده للمصرية Tier 2: {rules['sample_texts']['arabic_script']}",
    rules["sample_texts"]["tier2_academic"],
    "full_text_conversion",
    "tier2-rules.json sample_texts",
)

# ---------------------------------------------------------------------------
# 8. Loanword phonology micro-rules called out in the system prompt but not
#    present as structured JSON — hand-encoded here since they're small and
#    high-value (epenthesis, French/Greek é, cinema/cima judgment call).
# ---------------------------------------------------------------------------
extra_prompt_examples = [
    ("حوّل دي للمصرية: أسانسير", "asansēr", "loanword_egyptianized_french"),
    ("حوّل دي للمصرية: طرابيزة", "ṭarabéza", "loanword_egyptianized_greek"),
    ("حوّل دي للمصرية: كلاكس", "kalaks", "loanword_epenthesis"),
    ("حوّل دي للمصرية: بيانو", "piano", "loanword_visual_spelling"),
    ("حوّل دي للمصرية: سينما (استخدم أي شكل مقبول)", "cinema", "loanword_judgment_call"),
]
for u, a, cat in extra_prompt_examples:
    add(u, a, cat, "masri_tier2_system_prompt.md")

# ---------------------------------------------------------------------------
# Shuffle, split off a small in-train dev slice, write files
# ---------------------------------------------------------------------------
random.shuffle(examples)
n_dev = max(20, int(0.05 * len(examples)))
dev = examples[:n_dev]
train = examples[n_dev:]

with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

with open(OUT / "dev.jsonl", "w", encoding="utf-8") as f:
    for ex in dev:
        f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Eval set: derived STRICTLY from masri_tier2_eval_set.json.
# Kept as a SEPARATE file, never merged into train.jsonl, so pass-rate numbers
# mean something. Two rows per item when a franco variant exists.
# ---------------------------------------------------------------------------
eval_rows = []
for item in eval_set["items"]:
    base = {
        "id": item["id"],
        "category": item["category"],
        "expected": item["expected"],
        "accepted_variants": item.get("accepted_variants", []),
        "tests_for": item["tests_for"],
    }
    row = dict(base)
    row["input"] = item["arabic_script"]
    row["input_type"] = "arabic_script"
    eval_rows.append(row)
    if item.get("franco"):
        row2 = dict(base)
        row2["input"] = item["franco"]
        row2["input_type"] = "franco"
        eval_rows.append(row2)

with open(OUT / "eval_held_out.jsonl", "w", encoding="utf-8") as f:
    for row in eval_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# also save the system prompt alongside the data, since eval/inference need it
with open(OUT / "system_prompt.txt", "w", encoding="utf-8") as f:
    f.write(SYSTEM_PROMPT)

# stats
from collections import Counter

stats = {
    "train_examples": len(train),
    "dev_examples": len(dev),
    "eval_examples": len(eval_rows),
    "train_by_category": dict(Counter(e["category"] for e in train)),
    "eval_by_category": dict(Counter(r["category"] for r in eval_rows)),
}
with open(OUT / "dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False, indent=2))
