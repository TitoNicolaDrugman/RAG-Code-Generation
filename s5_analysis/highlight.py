import html
import re
from IPython.display import HTML, display

def highlight_tokens_html(text: str, query_tokens):
    if not text:
        display(HTML("<pre>(empty)</pre>")); return
    safe = html.escape(text)
    unique_qt = sorted(set([t for t in query_tokens if t]), key=len, reverse=True)
    # placeholding to avoid nested replacements
    for i, tkn in enumerate(unique_qt):
        safe = re.sub(rf"\b({re.escape(tkn)})\b", f"__HL_S5_{i}__", safe, flags=re.IGNORECASE)
    for i, tkn in enumerate(unique_qt):
        safe = safe.replace(
            f"__HL_S5_{i}__",
            f"<b style='background-color:#FFFACD; color:black; font-weight:bold;'>{html.escape(tkn)}</b>"
        )
    display(HTML(f"<pre style='white-space:pre-wrap; word-wrap:break-word; border:1px dashed #ccc; padding:6px; margin-left:20px;'>{safe}</pre>"))
import html
import re
from IPython.display import HTML, display

def highlight_tokens_html(text: str, query_tokens):
    if not text:
        display(HTML("<pre>(empty)</pre>")); return
    safe = html.escape(text)
    unique_qt = sorted(set([t for t in query_tokens if t]), key=len, reverse=True)
    # placeholding to avoid nested replacements
    for i, tkn in enumerate(unique_qt):
        safe = re.sub(rf"\b({re.escape(tkn)})\b", f"__HL_S5_{i}__", safe, flags=re.IGNORECASE)
    for i, tkn in enumerate(unique_qt):
        safe = safe.replace(
            f"__HL_S5_{i}__",
            f"<b style='background-color:#FFFACD; color:black; font-weight:bold;'>{html.escape(tkn)}</b>"
        )
    display(HTML(f"<pre style='white-space:pre-wrap; word-wrap:break-word; border:1px dashed #ccc; padding:6px; margin-left:20px;'>{safe}</pre>"))
