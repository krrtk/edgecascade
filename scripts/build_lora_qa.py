import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "training" / "lora"
OUTPUT_FILE = OUTPUT_DIR / "lora_qa_sft.jsonl"

def generate_qa_pairs():
    pairs = [
        # ISRO / Remote Sensing Q&A
        {
            "q": "What is remote sensing?",
            "a": "Remote sensing is the science and art of obtaining information about an object, area, or phenomenon through the analysis of data acquired by a device that is not in contact with the object, area, or phenomenon under investigation."
        },
        {
            "q": "How does electromagnetic radiation interact with the Earth's surface in remote sensing?",
            "a": "When electromagnetic radiation strikes the Earth's surface, it can be transmitted, absorbed, or reflected. The proportion of each depends on the surface material and the wavelength of the energy."
        },
        {
            "q": "What are the typical platforms used in remote sensing?",
            "a": "Typical platforms used in remote sensing include satellites, aircraft, drones (UAVs), and balloons. They carry the sensors that capture data from the Earth's surface."
        },
        {
            "q": "What is spatial resolution in the context of remote sensing?",
            "a": "Spatial resolution refers to the smallest object that can be resolved on the ground. In a digital image, it represents the area on the ground covered by a single pixel."
        },
        {
            "q": "What is the difference between active and passive remote sensing?",
            "a": "Passive remote sensing systems measure naturally available energy, such as sunlight reflected from the Earth. Active remote sensing systems provide their own energy source for illumination, such as radar."
        },
        {
            "q": "What are land observation satellites primarily used for?",
            "a": "Land observation satellites are primarily used to monitor the Earth's surface for applications like agriculture, forestry, water resources, urban planning, and disaster management."
        },
        {
            "q": "Why is the electromagnetic spectrum important for remote sensing?",
            "a": "The electromagnetic spectrum is critical because different Earth surface features reflect and absorb different wavelengths uniquely, allowing sensors to identify and distinguish between materials based on their spectral signatures."
        },
        {
            "q": "What is Geographic Information System (GIS)?",
            "a": "GIS is a computer-based system designed to capture, store, manipulate, analyze, manage, and present all types of geographical data."
        },
        {
            "q": "What is a vector data model in GIS?",
            "a": "A vector data model represents geographic features using points, lines, and polygons, defined by precise X,Y coordinates."
        },
        {
            "q": "What is a raster data model in GIS?",
            "a": "A raster data model represents geographic features as a grid of cells or pixels, where each cell contains a value representing information, typically used for continuous data like elevation or temperature."
        },
        {
            "q": "What is the purpose of Indian Regional Navigational Satellite System (IRNSS)?",
            "a": "IRNSS, also known as NavIC, is an independent regional navigation satellite system developed by India to provide accurate position information service to users in India as well as the region extending up to 1500 km from its boundary."
        },
        {
            "q": "How is geospatial technology used in disaster management?",
            "a": "Geospatial technology is used for hazard mapping, vulnerability assessment, early warning, damage assessment, and planning relief operations during natural disasters like floods, earthquakes, and landslides."
        },
        {
            "q": "What role does remote sensing play in agriculture?",
            "a": "In agriculture, remote sensing is used for crop acreage estimation, yield forecasting, drought monitoring, crop health assessment, and precision farming."
        },
        {
            "q": "How is 3D mapping conducted from space?",
            "a": "3D mapping from space is often conducted using stereo imaging, where a satellite captures two images of the same area from different angles, allowing for the generation of Digital Elevation Models (DEMs)."
        },
        {
            "q": "What is the Vegetation Condition Index (VCI) used for?",
            "a": "VCI is an indicator used to assess the status of vegetation health compared to historical norms, which is particularly useful for identifying and monitoring agricultural droughts."
        },

        # DPDPA Q&A
        {
            "q": "What is the primary purpose of the Digital Personal Data Protection Act (DPDPA)?",
            "a": "The primary purpose of the DPDPA is to provide for the processing of digital personal data in a manner that recognizes both the right of individuals to protect their personal data and the need to process such personal data for lawful purposes."
        },
        {
            "q": "Who is a Data Principal under the DPDPA?",
            "a": "A Data Principal is the individual to whom the personal data relates. In the case of a child or a person with a disability, it includes their parent or lawful guardian."
        },
        {
            "q": "What is the definition of a Data Fiduciary?",
            "a": "A Data Fiduciary is any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data."
        },
        {
            "q": "What does processing mean under the DPDPA?",
            "a": "Processing means a wholly or partly automated operation or set of operations performed on digital personal data, and includes operations such as collection, recording, organisation, structuring, storage, adaptation, retrieval, use, alignment or combination, indexing, sharing, disclosure by transmission, dissemination or otherwise making available, restriction, erasure or destruction."
        },
        {
            "q": "What are the requirements for valid consent under the DPDPA?",
            "a": "Valid consent must be free, specific, informed, unconditional and unambiguous with a clear affirmative action, and shall signify an agreement to the processing of her personal data for the specified purpose and be limited to such personal data as is necessary for such specified purpose."
        },
        {
            "q": "What rights does a Data Principal have regarding their data?",
            "a": "A Data Principal has the right to access information about personal data, the right to correction and erasure of personal data, the right of grievance redressal, and the right to nominate a person in case of death or incapacity."
        },
        {
            "q": "What is a Consent Manager?",
            "a": "A Consent Manager is a person registered with the Board, who acts as a single point of contact to enable a Data Principal to give, manage, review and withdraw her consent through an accessible, transparent and interoperable platform."
        },
        {
            "q": "Are Data Fiduciaries required to take security measures?",
            "a": "Yes, a Data Fiduciary is required to implement reasonable security safeguards to prevent personal data breach."
        },
        {
            "q": "What obligations does a Data Fiduciary have in the event of a data breach?",
            "a": "In the event of a personal data breach, the Data Fiduciary must intimate the Data Protection Board of India and each affected Data Principal."
        },
        {
            "q": "Can a Data Principal withdraw their consent?",
            "a": "Yes, a Data Principal has the right to withdraw their consent at any time, with the ease of doing so being comparable to the ease with which such consent was given."
        },
        {
            "q": "What is the role of the Data Protection Board of India?",
            "a": "The Board is established to direct remediation or mitigation in response to data breaches, inquire into breaches, impose penalties for non-compliance, and handle complaints from Data Principals."
        },
        {
            "q": "Does the DPDPA apply to non-digital personal data?",
            "a": "The DPDPA applies to personal data collected in digital form and personal data collected in non-digital form and digitized subsequently. It does not apply to non-digital data that remains non-digital."
        },
        {
            "q": "What is a Significant Data Fiduciary?",
            "a": "A Significant Data Fiduciary is a Data Fiduciary notified by the Central Government based on an assessment of factors like the volume and sensitivity of personal data processed, risk to the rights of Data Principals, potential impact on the sovereignty and integrity of India, risk to electoral democracy, security of the State, and public order."
        },
        {
            "q": "What additional obligations apply to a Significant Data Fiduciary?",
            "a": "Significant Data Fiduciaries must appoint a Data Protection Officer, appoint an independent data auditor, and undertake periodic Data Protection Impact Assessments."
        },
        {
            "q": "What is the mechanism for grievance redressal?",
            "a": "A Data Principal has the right to have grievances readily resolved by the Data Fiduciary or Consent Manager. If unsatisfied, they can register a complaint with the Data Protection Board of India."
        }
    ]
    return pairs

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pairs = generate_qa_pairs()
    
    total_tokens = 0
    total_q_len = 0
    total_a_len = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            q = p["q"]
            a = p["a"]
            
            # Format
            text = f"### Question:\n{q}\n\n### Answer:\n{a}"
            
            # Save
            record = {"text": text}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            total_q_len += len(q.split())
            total_a_len += len(a.split())
            total_tokens += len(text.split()) * 1.3 # Rough token estimation
            
    print(f"LoRA QA SFT dataset built at {OUTPUT_FILE}")
    print(f"Total examples: {len(pairs)}")
    print(f"Avg question length: {total_q_len / len(pairs):.1f} words")
    print(f"Avg answer length: {total_a_len / len(pairs):.1f} words")
    print(f"Estimated tokens: {int(total_tokens)}")

if __name__ == "__main__":
    main()
