"""Dedupe, cap per document, label S1..Sn and build the evidence block."""
from __future__ import annotations

import config
from core.schemas import EvidenceItem


def assemble(scored_chunks: list[dict], limit: int = config.EVIDENCE_BLOCK_SIZE,
             max_per_doc: int = config.MAX_CHUNKS_PER_DOC) -> list[EvidenceItem]:
    """Pick a diverse, capped set of chunks and label them S1..Sn."""
    per_doc: dict[str, int] = {}
    seen: set[str] = set()
    picked: list[dict] = []

    for chunk in scored_chunks:
        chunk_id = chunk["_id"]
        doc_id = chunk.get("doc_id", "")
        if chunk_id in seen:
            continue
        if per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        seen.add(chunk_id)
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        picked.append(chunk)
        if len(picked) >= limit:
            break

    return [
        EvidenceItem(
            label=f"S{i}",
            chunk_id=chunk["_id"],
            doc_id=chunk.get("doc_id", ""),
            doc_title=chunk.get("doc_title", ""),
            text=chunk.get("text", ""),
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            score=round(float(chunk.get("final_score", 0.0)), 4),
            evidence_level=chunk.get("evidence_level"),
            content_role=chunk.get("content_role"),
            climate_zones=chunk.get("climate_zones") or [],
            practices=chunk.get("practices") or [],
            claims=chunk.get("claims") or [],
        )
        for i, chunk in enumerate(picked, start=1)
    ]


def evidence_map(items: list[EvidenceItem]) -> dict[str, EvidenceItem]:
    """S# -> item, used by verification and by the citation renderer."""
    return {item.label: item for item in items}


def render_evidence_block(items: list[EvidenceItem]) -> str:
    """The labelled evidence text handed to the reasoning call."""
    parts = []
    for item in items:
        claims = []
        for claim in item.claims:
            value = claim.get("value")
            if value is None:
                continue
            claims.append(
                f"{claim.get('metric')} {claim.get('direction')} {value}"
                f"{' ' + claim.get('unit') if claim.get('unit') else ''}"
                f"{' [' + str(claim.get('conditions')) + ']' if claim.get('conditions') else ''}"
            )
        header = (
            f"[{item.label}] {item.doc_title} (p{item.page_start}) "
            f"role={item.content_role} evidence={item.evidence_level} "
            f"zones={','.join(item.climate_zones) or 'unstated'} "
            f"practices={','.join(item.practices) or 'none'}"
        )
        body = item.text.strip()
        claim_line = f"\nCLAIMS: {'; '.join(claims)}" if claims else "\nCLAIMS: none"
        parts.append(f"{header}\n{body}{claim_line}")
    return "\n\n".join(parts)
