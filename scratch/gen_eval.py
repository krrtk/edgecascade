import json
from pathlib import Path

isro_path = 'data/evaluation/isro/cartosat1_holdout.json'
dpdpa_path = 'data/evaluation/dpdpa/dpdpa_holdout.json'

with open(isro_path, 'r', encoding='utf-8') as f:
    isro_data = json.load(f)
with open(dpdpa_path, 'r', encoding='utf-8') as f:
    dpdpa_data = json.load(f)

def find_seg(data, query):
    segs = [x['segment_id'] for x in data if query.lower() in x['text'].lower()]
    return segs

q_defs = [
    # ISRO Knowledge Gap
    {"q": "What is the swath of Cartosat-1 in stereo mode?", "query": "stereo mode", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The swath of Cartosat-1 in stereo mode is 26 km."},
    {"q": "What is the OBSSR capacity?", "query": "OBSSR", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The OBSSR capacity is 120 GB."},
    {"q": "What is the repeat cycle of the Cartosat-1 orbit?", "query": "repeat cycle", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The repeat cycle is 116 days."},
    {"q": "What are the two cameras used for stereo imaging?", "query": "PAN", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The two cameras are PAN-Fore and PAN-Aft."},
    {"q": "What is the fixed base-to-height ratio for stereo pairs?", "query": "base to height", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The fixed base-to-height ratio is 0.62."},
    {"q": "When was Cartosat-1 launched?", "query": "May 05,2005", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "Cartosat-1 was launched on May 05, 2005."},
    {"q": "What is the spatial resolution of the PAN sensors?", "query": "2.5m", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The PAN sensors have a 2.5m resolution."},
    {"q": "What orbit altitude does Cartosat-1 have?", "query": "altitude", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The altitude is 618 km."},
    {"q": "What is the local time of equator crossing?", "query": "Equatorial crossing time", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The local time of equator crossing is 10:30 AM."},
    {"q": "What is the payload data rate?", "query": "data rate", "domain": "isro", "cat": "knowledge_gap", "tier": "rag", "ans": "The payload data rate is 105 Mbps per camera."},
    
    # ISRO Reasoning
    {"q": "How does the combination of the two PAN cameras enable 3D terrain modeling?", "query": "stereo capability", "domain": "isro", "cat": "reasoning", "tier": "remote", "ans": "The PAN-Fore and PAN-Aft cameras capture images at different angles (+26 and -5 degrees), providing fore-aft stereo capability which is necessary for generating Digital Elevation Models (DEMs)."},
    {"q": "Why is the 10:30 AM equator crossing time significant for Cartosat-1's optical sensors?", "query": "Equatorial crossing time", "domain": "isro", "cat": "reasoning", "tier": "remote", "ans": "The 10:30 AM crossing provides optimal sun illumination angles for optical imaging, reducing shadows while highlighting terrain features."},
    {"q": "What is the advantage of using an OBSSR over a traditional tape recorder?", "query": "Solid State Recorder", "domain": "isro", "cat": "reasoning", "tier": "remote", "ans": "The On-Board Solid State Recorder (OBSSR) has no moving parts, higher reliability, and allows simultaneous recording and playback compared to magnetic tape recorders."},
    {"q": "How does Cartosat-1's stereo swath compare to its mono mode swath?", "query": "swath", "domain": "isro", "cat": "reasoning", "tier": "remote", "ans": "In stereo mode, the swath is 26 km, whereas in mono mode (Wide Mono), the combined swath of both cameras can be up to 30 km."},
    {"q": "Why does Cartosat-1 need both X-band and S-band communications?", "query": "communication", "domain": "isro", "cat": "reasoning", "tier": "remote", "ans": "X-band is used for high-speed payload data downlink (like imagery), while S-band is used for low-speed Telemetry, Tracking, and Command (TTC) operations."},

    # DPDPA Knowledge Gap
    {"q": "What conditions must consent satisfy?", "query": "clear affirmative action", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "Consent must be free, specific, informed, unconditional and unambiguous with a clear affirmative action."},
    {"q": "What right does a Data Principal have regarding previously given consent?", "query": "withdraw her consent", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "The Data Principal has the right to withdraw her consent at any time."},
    {"q": "What must happen after withdrawal of consent?", "query": "withdraws her consent", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "The Data Fiduciary and its Data Processors must cease processing the personal data within a reasonable time."},
    {"q": "What factors must the Board consider when determining a monetary penalty?", "query": "monetary penalty to be imposed", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "The Board must consider the nature, gravity and duration of the breach, the type of data affected, repetitive nature, any gain or loss avoided, mitigation actions taken, proportionality, and impact of the penalty."},
    {"q": "What does Section 33 say about monetary penalties?", "query": "monetary penalty", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "Section 33 allows the Board to impose monetary penalties specified in the Schedule after giving the person an opportunity of being heard, if a significant breach is determined."},
    {"q": "Can a Data Principal manage consent through a Consent Manager?", "query": "Consent Manager", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "Yes, a Data Principal may give, manage, review, or withdraw consent through a Consent Manager."},
    {"q": "What language requirements exist for consent requests?", "query": "Eighth Schedule", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "Requests must be in clear and plain language with an option to access them in English or any language specified in the Eighth Schedule to the Constitution."},
    {"q": "What is the maximum penalty for breach of obligation by a Data Fiduciary to take reasonable security safeguards?", "query": "crore rupees", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "The penalty may extend to two hundred and fifty crore rupees."},
    {"q": "Who can amend the Schedule of penalties?", "query": "amend the Schedule", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "The Central Government may, by notification, amend the Schedule."},
    {"q": "Does withdrawal of consent affect the legality of prior processing?", "query": "legality of processing", "domain": "dpdpa", "cat": "knowledge_gap", "tier": "rag", "ans": "No, such withdrawal shall not affect the legality of processing of the personal data based on consent before its withdrawal."},

    # DPDPA Reasoning
    {"q": "How does the DPDPA balance the rights of the Data Principal with the practical needs of the Data Fiduciary regarding consent withdrawal?", "query": "withdraw", "domain": "dpdpa", "cat": "reasoning", "tier": "remote", "ans": "While the Data Principal can withdraw consent easily (Sec 6(4)), the consequences are borne by the Principal, prior processing remains legal (Sec 6(5)), and the Fiduciary has a 'reasonable time' to cease processing (Sec 6(6))."},
    {"q": "Why is it important that a Consent Manager is registered with the Board?", "query": "accountable to the Data Principal", "domain": "dpdpa", "cat": "reasoning", "tier": "remote", "ans": "Registration ensures the Consent Manager meets technical, operational, and financial conditions (Sec 6(9)) because they act on behalf of the Data Principal and are accountable to them (Sec 6(8))."},
    {"q": "In a dispute over whether consent was given, who bears the burden of proof and how do they prove it?", "query": "obliged to prove", "domain": "dpdpa", "cat": "reasoning", "tier": "remote", "ans": "The Data Fiduciary bears the burden of proof (Sec 6(10)) and must prove that a notice was given and consent was obtained in accordance with the Act."},
    {"q": "How do the criteria for determining a monetary penalty ensure fairness and proportionality?", "query": "determining the amount of monetary penalty", "domain": "dpdpa", "cat": "reasoning", "tier": "remote", "ans": "Section 33(2) mandates looking at mitigating actions, repetitive nature, and whether a gain was realised, ensuring the penalty is proportionate to the severity and the entity's behavior."},
    {"q": "What limitation exists on the Central Government's power to amend the penalties in the Schedule?", "query": "twice of what was specified", "domain": "dpdpa", "cat": "reasoning", "tier": "remote", "ans": "Section 42(1) restricts the Central Government from increasing any penalty to more than twice of what was originally specified in the Act."},
]

out_records = []
for i, q in enumerate(q_defs):
    data = isro_data if q['domain'] == 'isro' else dpdpa_data
    segs = find_seg(data, q['query'])
    
    if not segs:
        print(f"FAILED TO FIND SEGMENTS FOR: {q['q']} (Query: {q['query']})")
        segs = [data[0]['segment_id']] # Fallback just in case
        
    out = {
        "question_id": f"{q['domain']}_eval_{i:03d}",
        "domain": q['domain'],
        "category": q['cat'],
        "question": q['q'],
        "ground_truth": q['ans'],
        "source_segments": segs[:3],
        "expected_tier": q['tier'],
        "source_doc": "holdout_document.pdf",
        "section_ref": "various",
        "notes": "Generated for first evaluation"
    }
    out_records.append(out)

Path('data/evaluation').mkdir(parents=True, exist_ok=True)
with open('data/evaluation/evaluation_questions.jsonl', 'w', encoding='utf-8') as f:
    for rec in out_records:
        f.write(json.dumps(rec) + '\n')

print(f"Generated {len(out_records)} questions.")
