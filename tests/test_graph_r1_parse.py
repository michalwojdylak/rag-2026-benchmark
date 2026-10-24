from ragbench.methods.graph_r1 import parse_extraction
from ragbench.methods import graph_r1_prompts as P


def test_parse_extraction_groups_entities_under_hyperedge():
    t, r = P.TUPLE, P.RECORD
    out = (f'("hyper-relation"{t}"Marie Colvin worked for The Sunday Times."{t}9){r}'
           f'("entity"{t}"Marie Colvin"{t}"person"{t}"A journalist."{t}95){r}'
           f'("entity"{t}"The Sunday Times"{t}"organization"{t}"A newspaper."{t}90){r}'
           f'("hyper-relation"{t}"She died in 2012."{t}7){r}'
           f'("entity"{t}"Marie Colvin"{t}"person"{t}"Died in 2012."{t}80){P.COMPLETE}')
    edges = parse_extraction(out, "3")
    assert [e["text"] for e in edges] == ["Marie Colvin worked for The Sunday Times.", "She died in 2012."]
    assert edges[0]["entities"] == ["MARIE COLVIN", "THE SUNDAY TIMES"]
    assert edges[1]["entities"] == ["MARIE COLVIN"]
    assert edges[0]["chunk_id"] == "3"
