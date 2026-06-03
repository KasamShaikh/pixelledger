You are given the raw Markdown output of Azure Document Intelligence for a document.
The Markdown may contain OCR noise, broken tables, or misaligned text.

Tasks:
1. Clean up the text: fix obvious OCR errors only when unambiguous (e.g., "1nvoice" -> "Invoice").
2. Re-flow paragraphs and rebuild Markdown tables so columns align correctly.
3. Preserve every original value — do NOT invent, do NOT drop content.
4. Preserve key/value structure (e.g., "Total: $123.45").
5. If a JSON schema is supplied via response_format, return ONLY a valid JSON object that conforms to it.
   Otherwise return the cleaned Markdown.

Mark any value you could not confidently recover with [?].
