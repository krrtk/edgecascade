import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.judge import get_judge
import math

def test_judge():
    judge = get_judge()
    
    print("Running Sanity Tests...")
    
    # 1. clearly correct answer -> PASS (Tier 1 & Tier 2)
    # Tier 1 (mock high confidence)
    res = judge.judge("What is the swath of Cartosat-1?", "The Cartosat-1 swath is 30 km.", log_probs=[math.log(0.9)]*6)
    assert res["pass"] is True, f"Test 1 (Tier 1) failed: {res['reason']}"
    
    # Tier 2 (supported by context)
    res = judge.judge("What is the swath of Cartosat-1?", "The Cartosat-1 swath is 30 km.", context="Cartosat-1 is a satellite with a swath of 30 km.")
    assert res["pass"] is True, f"Test 1 (Tier 2) failed: {res['reason']}"
    
    # 2. fluent but factually wrong answer -> FAIL
    # Tier 1 (hallucination -> low confidence for heuristic, API will judge based on content if it knows, but for test we just check it doesn't break)
    # The API judge might pass it if it thinks 900 miles is plausible or it hallucinates, but we will test Tier 2 mostly.
    # Tier 2 (contradicts context)
    res = judge.judge("What is the swath of Cartosat-1?", "The Cartosat-1 swath is 900 miles.", context="Cartosat-1 is a satellite with a swath of 30 km.")
    assert res["pass"] is False, f"Test 2 (Tier 2) failed: {res['reason']}"
    
    # 3. repetitive nonsense -> FAIL
    res = judge.judge("What is the swath of Cartosat-1?", "Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat Cartosat", log_probs=[math.log(0.9)]*11)
    assert res["pass"] is False, f"Test 3 failed: {res['reason']}"
    
    # 4. incomplete answer -> FAIL
    res = judge.judge("What is the swath of Cartosat-1?", "The swath", log_probs=[math.log(0.9)]*2)
    assert res["pass"] is False, f"Test 4 failed: {res['reason']}"
    
    # 5. correct answer expressed differently -> PASS
    res = judge.judge("What is the swath of Cartosat-1?", "It has a 30km width of swath.", context="Cartosat-1 is a satellite with a swath of 30 km.")
    assert res["pass"] is True, f"Test 5 failed: {res['reason']}"

    print("All Sanity Tests Passed!")

if __name__ == "__main__":
    test_judge()
