import re
import json
import os
from pathlib import Path
from typing import Any
import requests
import streamlit as st

# Setup Default Path for Tender Reports
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs" / "tender_reports"

class VllmChatClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        base_url: str = "http://10.4.25.56:8000/v1",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        body = resp.json() or {}
        choices = body.get("choices") or []
        content: Any = None
        if isinstance(choices, list) and choices:
            content = ((choices[0] or {}).get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("VLLM returned empty content")
            
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        if '<think>' in content:
            content = re.sub(r'<think>.*', '', content, flags=re.DOTALL).strip()
            
        return content

@st.cache_data
def load_tenders():
    tenders = []
    if OUTPUTS_DIR.exists():
        for d in sorted(OUTPUTS_DIR.iterdir()):
            if d.is_dir():
                json_path = d / f"{d.name}_document_1.json"
                if not json_path.exists():
                    json_path = d / f"{d.name}_report.json"
                
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        # Extract basic info for quick display
                        overview = data.get("document_analysis", {}).get("tender_overview", {})
                        if not overview:
                            # Fallback if document_analysis isn't top-level
                            overview = data.get("tender_overview", {})

                        title = overview.get("tender_title", f"Tender {d.name}")
                        
                        short_title = title
                        try:
                            llm = VllmChatClient()
                            msgs = [
                                {"role": "system", "content": "You are a helpful assistant. Provide a very short, concise title (max 5 words) for the given tender. Do NOT add quotes. Do NOT add prefixes."},
                                {"role": "user", "content": f"Tender Title: {title}"}
                            ]
                            resp = llm.chat(msgs).strip()
                            resp = resp.replace('"', '').replace("'", "")
                            if resp.lower().startswith("title:"): resp = resp[6:].strip()
                            if len(resp) > 60: resp = resp[:57] + "..."
                            if len(resp) > 3: short_title = resp
                        except Exception:
                            pass
                        
                        doc2_data = None
                        doc3_data = None
                        doc2_path = d / f"{d.name}_document_2.json"
                        if doc2_path.exists():
                            with open(doc2_path, "r", encoding="utf-8") as f2:
                                doc2_data = json.load(f2)
                        
                        doc3_path = d / f"{d.name}_document_3.json"
                        if doc3_path.exists():
                            with open(doc3_path, "r", encoding="utf-8") as f3:
                                doc3_data = json.load(f3)

                        tenders.append({
                            "id": d.name,
                            "title": short_title,
                            "original_title": title,
                            "data": data,
                            "doc2": doc2_data,
                            "doc3": doc3_data,
                            "path": str(json_path),
                            "raw_text": json.dumps(data).lower() + json.dumps(doc2_data or {}).lower() + json.dumps(doc3_data or {}).lower()
                        })
                    except Exception:
                        pass
    return tenders

def extract_numeric(val_str: str) -> float:
    if not isinstance(val_str, str): return 0.0
    match = re.search(r'([\d,]{4,}(?:\.\d+)?)', val_str.replace(" ", ""))
    if match:
        clean = match.group(1).replace(",", "")
        try:
            return float(clean)
        except ValueError:
            pass
    return 0.0

def format_currency(val: float) -> str:
    if val == 0.0: return "N/A"
    return f"₹{val:,.2f}"

def render_tender_view(tender: dict):
    st.markdown(f"## {tender.get('original_title', tender['title'])}")
    st.caption(f"Tender ID: {tender['id']}")
    
    data = tender["data"]
    doc_an = data.get("document_analysis", data)
    overview = doc_an.get("tender_overview", {})
    mt = doc_an.get("money_and_timeline", {})
    aw = doc_an.get("award_information", {})
    
    est_val_str = str(mt.get("estimated_contract_value", "0"))
    aw_val_str = str(aw.get("awarded_amount", "0"))
    
    est_val = extract_numeric(est_val_str)
    aw_val = extract_numeric(aw_val_str)
    
    delta = None
    delta_str = None
    if est_val > 0 and aw_val > 0:
        delta = aw_val - est_val
        delta_pct = (delta / est_val) * 100
        delta_str = f"₹{delta:,.2f} ({delta_pct:+.2f}%)"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_color = "normal" if overview.get("status", "").lower() == "awarded" else "off"
        st.metric("Status", overview.get("status", "N/A").upper())
    with col2:
        st.metric("Expected Contract Value", format_currency(est_val) if est_val > 0 else est_val_str)
    with col3:
        st.metric("Actual Awarded Value", format_currency(aw_val) if aw_val > 0 else aw_val_str, delta=delta_str, delta_color="inverse")
    with col4:
        st.metric("Awarded To", str(aw.get('awarded_to', 'N/A')))
        
    st.divider()
    
    tab_summary, tab_details, tab_sources, tab_vendor, tab_json = st.tabs(["Dashboard", "Details", "Web Sources", "Vendor", "Raw JSON"])
    
    with tab_summary:
        c1, c2 = st.columns([2, 1])
        with c1:
            smr = doc_an.get("citizen_friendly_summary", [])
            st.markdown("### Citizen Friendly Summary")
            if smr:
                for point in smr:
                    st.success(point)
            else:
                st.info("No summary available.")
                
        with c2:
            flags = doc_an.get("red_flags", [])
            st.markdown("### Red Flags identified")
            if flags:
                for f in flags:
                    st.error(f)
            else:
                st.write("No major red flags.")
                
            st.markdown("### Authority Info")
            st.info(f"**Issued By**:\n{overview.get('issuing_authority', 'N/A')}\n\n**Published Date**:\n{overview.get('publication_date', 'N/A')}\n\n**Closing Date**:\n{overview.get('closing_date_time', 'N/A')}")

    with tab_details:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Money & Timeline")
            for k, v in mt.items():
                st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")
                
        with c2:
            st.markdown("### Award Information")
            for k, v in aw.items():
                st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")
                
        st.markdown("### Bidding & Eligibility")
        be = doc_an.get("bidding_and_eligibility", {})
        for k, v in be.items():
            st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")

    with tab_sources:
        doc2 = tender.get("doc2")
        if doc2:
            overall = doc2.get("overall_summary") or doc2.get("summary") or "No background context found."
            st.markdown(f"### Web Overview\n{overall}")
            
            tender_spec = doc2.get("tender_specific_news")
            if tender_spec:
                st.markdown(f"**Tender Specific News:** {tender_spec}")
            
            market = doc2.get("market_and_location_context")
            if market:
                st.markdown(f"**Market Context:** {market}")

            sources = doc2.get("sources", [])
            if sources:
                st.markdown("### Cited Sources")
                for src in sources:
                    st.markdown(f"- [{src.get('title', 'Link')}]({src.get('url', '#')}) (Relevance: {src.get('relevance')})")
        else:
            st.info("No web sources (document_2) found.")
            
    with tab_vendor:
        doc3 = tender.get("doc3")
        if doc3:
            st.markdown(f"### Vendor: {doc3.get('vendor_name', 'Unknown')}")
            
            bp = doc3.get("business_profile", {})
            st.write(f"**Status**: {bp.get('status', 'N/A')}")
            st.write(f"**Registered Name**: {bp.get('registered_name', 'N/A')}")
            dirs = bp.get("directors_or_proprietor", [])
            if dirs:
                st.write(f"**Directors**: {', '.join(dirs)}")
                
            ids = doc3.get("identifiers", {})
            st.markdown("**Identifiers:**")
            st.write(ids)
            
            flags = doc3.get("risk_flags", [])
            if flags:
                st.markdown("### Vendor Risk Flags")
                for fg in flags:
                    st.warning(f"- {fg}")
        else:
            st.info("No vendor profile (document_3) found.")

    with tab_json:
        st.markdown("### Raw Pipeline Output Sections")
        
        with st.expander("Document 1: Tender Report Analysis", expanded=True):
            st.json(data)
        with st.expander("Document 2: Web News Context", expanded=False):
            st.json(tender.get("doc2") or {})
        with st.expander("Document 3: Vendor Background", expanded=False):
            st.json(tender.get("doc3") or {})

def main():
    st.set_page_config(layout="wide", page_title="Tender Report Hub")
    
    tenders = load_tenders()
    if not tenders:
        st.warning(f"No tender reports found in `{OUTPUTS_DIR}`.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("OpenTender Hyderabad")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        mode = st.selectbox("Navigation", ["Browse & Search", "Compare Tenders", "LLM Chat Assistant"], label_visibility="collapsed")

    if mode == "Browse & Search":
        st.sidebar.title("Search Filters")
        search_query = st.sidebar.text_input("Search across tender documents (keyword):")
        
        filtered = tenders
        if search_query:
            filtered = [
                t for t in tenders 
                if search_query.lower() in t["raw_text"]
            ]

        st.sidebar.info(f"Found {len(filtered)} tenders matching your criteria.")
        
        if filtered:
            st.sidebar.markdown("### Available Tenders")
            selected_id = st.sidebar.radio(
                "Select Tender Header", 
                options=[t['id'] for t in filtered],
                format_func=lambda x: "» " + next((t['title'] for t in filtered if t['id'] == x), x),
                label_visibility="collapsed"
            )
            selected_tender = next((t for t in filtered if t['id'] == selected_id), None)
            if selected_tender:
                st.divider()
                render_tender_view(selected_tender)
                
    elif mode == "Compare Tenders":
        st.header("Compare Tenders")
        
        if len(tenders) < 2:
            st.warning("Not enough tenders to compare.")
            return

        tender_options = {t['id']: f"[{t['id']}] {t['title']}" for t in tenders}
        
        col1, col2 = st.columns(2)
        with col1:
            id1 = st.selectbox("Tender 1", options=list(tender_options.keys()), format_func=lambda x: tender_options[x], index=0)
        with col2:
            id2 = st.selectbox("Tender 2", options=list(tender_options.keys()), format_func=lambda x: tender_options[x], index=min(1, len(tenders) - 1))
            
        st.divider()
        t1 = next(t for t in tenders if t['id'] == id1)
        t2 = next(t for t in tenders if t['id'] == id2)
        
        c1, c2 = st.columns(2)
        with c1:
            render_tender_view(t1)
        with c2:
            render_tender_view(t2)
                
    elif mode == "LLM Chat Assistant":
        st.header("Ask about the Tenders")
        
        selected_tender_ids = st.multiselect(
            "Select tenders to include as context for the LLM:", 
            options=[t['id'] for t in tenders],
            format_func=lambda x: f"[{x}] " + next(t['title'] for t in tenders if t['id'] == x)
        )
        
        question = st.text_area("Your Question:")
        
        if st.button("Ask Assistant", type="primary") and question:
            if not selected_tender_ids:
                st.error("Please select at least one tender for context.")
                return
                
            with st.spinner("Analyzing and retrieving answer from LLM..."):
                selected_tenders = [t for t in tenders if t['id'] in selected_tender_ids]
                context_str = ""
                for t in selected_tenders:
                    context_str += f"\n\n=== Tender {t['id']} ===\n"
                    # Pass the analysis structure to the LLM to save tokens
                    doc_an = t["data"].get("document_analysis", t["data"])
                    context_str += json.dumps(doc_an, indent=2)[:8000] 
                
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful expert analyzing government tenders. Use the provided tender context to answer the user's question accurately and concisely."
                    },
                    {
                        "role": "user", 
                        "content": f"Context:\n{context_str}\n\nQuestion: {question}"
                    }
                ]
                
                llm = VllmChatClient(model="Qwen/Qwen3-30B-A3B-GPTQ-Int4")
                try:
                    response = llm.chat(messages)
                    st.success("Analysis Complete")
                    st.markdown("### Answer")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error fetching response from LLM: {e}")

if __name__ == "__main__":
    main()
